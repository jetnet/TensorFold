//! TF_CUDA_HOST_WFE=1 (off by default): the named host waits park the core in WFET instead of polling.
const std = @import("std");
const builtin = @import("builtin");
const abi = @import("abi.zig");
const Error = @import("driver.zig").Error;
const memory = @import("memory.zig");
const Stream = @import("stream.zig").Stream;
const sys = @import("sys");

const arm = builtin.cpu.arch == .aarch64 and builtin.os.tag == .linux;
const hwcap2_wfxt: u64 = 1 << 31; // HWCAP2_WFXT in Linux's arch/arm64/include/uapi/asm/hwcap.h
const auxval = if (builtin.link_libc) std.c.getauxval else std.os.linux.getauxval;
pub const Mode = enum(u8) { unread, off, on, no_wfxt };
var mode = std.atomic.Value(Mode).init(.unread);
const Totals = struct { waits: u64 = 0, parks: u64 = 0, by_marker: u64 = 0, by_query: u64 = 0, ticks: u64 = 0 };
var totals_lock: std.atomic.Mutex = .unlocked;
var totals: Totals = .{};
var logged_at: u64 = 0;

/// What the setting and AT_HWCAP2 allow: on only when asked on an aarch64 Linux host with WFET.
pub fn modeOf(asked: bool, hwcap2: u64) Mode {
    if (!asked) return .off;
    return if (arm and hwcap2 & hwcap2_wfxt != 0) .on else .no_wfxt;
}
/// The setting, decided once per process; only the deciding caller logs.
pub fn on() bool {
    const m = mode.load(.acquire);
    if (m != .unread) return m == .on;
    const asked = sys.environment.read(bool, "TF_CUDA_HOST_WFE", false) catch false; // startup refuses bad text
    const v = modeOf(asked, if (arm) @as(u64, auxval(std.elf.AT.HWCAP2)) else 0);
    if (mode.cmpxchgStrong(.unread, v, .acq_rel, .acquire)) |won| return won == .on;
    switch (v) {
        .on => std.log.info("host waits: WFET parks of up to 1 ms (TF_CUDA_HOST_WFE=1)", .{}),
        .no_wfxt => std.log.warn("TF_CUDA_HOST_WFE=1 needs aarch64 Linux with FEAT_WFxT; waits keep polling", .{}),
        .unread, .off => {},
    }
    return v == .on;
}
fn now() u64 {
    return asm volatile ("isb\nmrs %[t], cntvct_el0"
        : [t] "=r" (-> u64),
        :
        : .{ .memory = true });
}
/// One millisecond in counter ticks.
fn slice() u64 {
    const f = asm volatile ("mrs %[f], cntfrq_el0"
        : [f] "=r" (-> u64),
    );
    return f / 1000;
}
/// Waits while `word` holds `old`, by Arm ARM DDI 0487 D1.7.1: clearing LDXR's monitor is a WFET wake-up event.
fn waitChange(word: *const u32, old: u32, deadline: u64) void {
    // SEVL then WFE empty the Event Register; LDXR arms the monitor; WFET (x9 = CNTVCT deadline) waits for an event
    asm volatile (
        \\sevl
        \\wfe
        \\ldxr w10, [x11]
        \\eor w10, w10, w12
        \\cbnz w10, 1f
        \\.inst 0xd5031009
        \\1:
        :
        : [p] "{x11}" (word),
          [old] "{x12}" (@as(u64, old)),
          [dl] "{x9}" (deadline),
        : .{ .x10 = true, .memory = true });
}
/// The RoCE proxy's idle wait on its doorbell: returns on a store to the word, any other event, or after 1 ms.
pub fn park(word: *const u32, old: u32) void {
    if (comptime !arm) return;
    waitChange(word, old, now() + slice());
}
/// Adds one finished marker wait to the totals; a debug line carries them at most every 10 s.
fn count(parks: u64, by_marker: bool, ticks: u64) void {
    while (!totals_lock.tryLock()) std.atomic.spinLoopHint();
    defer totals_lock.unlock();
    totals.waits += 1;
    totals.parks += parks;
    if (by_marker) totals.by_marker += 1 else totals.by_query += 1;
    totals.ticks += ticks;
    const t = now();
    if (totals.waits != 1 and t -% logged_at < 10_000 * slice()) return;
    logged_at = t;
    const ms = totals.ticks / @max(1, slice());
    std.log.debug("host waits: {d} marker waits, {d} parks, {d} ended by the marker, {d} by the query, {d} ms", .{
        totals.waits, totals.parks, totals.by_marker, totals.by_query, ms,
    });
}
/// One owner's mapped marker word, used by one thread at a time, allocated at its first parked wait.
pub const Waiter = struct {
    buf: ?memory.HostBuffer = null,
    dev: abi.DevicePtr = 0,
    seq: u32 = 0,
    refused: bool = false, // after any unexpected result this waiter takes the existing synchronize only
    pending: ?Stream = null, // a queued marker write not yet known to have landed
    pub fn deinit(w: *Waiter) void {
        if (w.pending) |s| s.synchronize() catch { // no proof that the GPU's write retired: keep its 64-byte word
            std.log.warn("TF_CUDA_HOST_WFE: a marker write may still land; its word is kept", .{});
            w.* = .{};
            return;
        };
        if (w.buf) |*b| b.free();
        w.* = .{};
    }
    /// `stream.synchronize()`; with the setting on, the core first parks until a marker queued after the work lands.
    pub fn synchronize(w: *Waiter, stream: Stream) Error!void {
        if (comptime arm) if (on() and !w.refused) w.parked(stream);
        try stream.synchronize(); // at once after a landed marker, the whole wait after a fallback
        w.pending = null;
    }
    fn refuse(w: *Waiter, what: []const u8, res: ?abi.Result) void {
        w.refused = true;
        std.log.warn("TF_CUDA_HOST_WFE: {s} failed ({?d}); this waiter uses the existing synchronize", .{ what, res });
    }
    /// Parks until the marker lands or the stream is done; anything else hands the wait back to the existing path.
    fn parked(w: *Waiter, s: Stream) void {
        const d = s.d;
        const write = d.write_value32 orelse return w.refuse("the cuStreamWriteValue32_v2 lookup", null);
        if (w.buf == null) {
            var b = memory.HostBuffer.allocMapped(d, 64) catch return w.refuse("the marker allocation", null);
            w.dev = b.device() catch {
                b.free();
                return w.refuse("the marker's device address", null);
            };
            @atomicStore(u32, @as(*u32, @ptrCast(b.bytes.ptr)), w.seq, .release);
            w.buf = b;
        }
        const word: *u32 = @ptrCast(w.buf.?.bytes.ptr);
        const next = w.seq +% 1;
        const r = write(s.handle, w.dev, next, 0); // flags 0: ordered after the stream's work, with a system fence
        if (r != abi.success) return w.refuse("cuStreamWriteValue32_v2", r);
        w.seq = next;
        w.pending = s;
        const tick = slice();
        const start = now();
        var query_at = start + tick; // one deadline across wakeups, so frequent events cannot postpone the query
        var parks: u64 = 0;
        while (true) {
            const v = @atomicLoad(u32, word, .acquire);
            if (v == next) return count(parks, true, now() - start);
            waitChange(word, v, query_at);
            parks += 1;
            if (now() < query_at) continue;
            const q = d.api.cuStreamQuery(s.handle);
            if (q == abi.success) return count(parks, false, now() - start); // done before the marker was seen
            if (q != abi.error_not_ready) return w.refuse("cuStreamQuery", q);
            query_at = now() + tick;
        }
    }
};

/// A driver for the host tests: counts marker writes, synchronizes and frees; NOT_READY stands for a failure.
const Fake = struct {
    var word: [64]u8 align(64) = @splat(0);
    var writes: usize = 0;
    var syncs: usize = 0;
    var frees: usize = 0;
    var queries: usize = 0;
    var gpu_writes = true; // the "GPU" stores the marker as it is queued
    var write_result: abi.Result = abi.success;
    var sync_result: abi.Result = abi.success;
    var query_results: []const abi.Result = &.{}; // then success
    fn reset() void {
        word = @splat(0);
        writes = 0;
        syncs = 0;
        frees = 0;
        queries = 0;
        gpu_writes = true;
        write_result = abi.success;
        sync_result = abi.success;
        query_results = &.{};
    }
    fn hostAlloc(p: *?*anyopaque, _: usize, _: c_uint) callconv(.c) abi.Result {
        p.* = &word;
        return abi.success;
    }
    fn devicePointer(dev: *abi.DevicePtr, _: ?*anyopaque, _: c_uint) callconv(.c) abi.Result {
        dev.* = 0x1000;
        return abi.success;
    }
    fn freeHost(_: ?*anyopaque) callconv(.c) abi.Result {
        frees += 1;
        return abi.success;
    }
    fn write(_: abi.Stream, _: abi.DevicePtr, v: u32, _: c_uint) callconv(.c) abi.Result {
        writes += 1;
        if (write_result == abi.success and gpu_writes) @atomicStore(u32, @as(*u32, @ptrCast(&word)), v, .release);
        return write_result;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        syncs += 1;
        return sync_result;
    }
    fn query(_: abi.Stream) callconv(.c) abi.Result {
        queries += 1;
        return if (queries <= query_results.len) query_results[queries - 1] else abi.success;
    }
    fn driver() @import("driver.zig").Driver {
        var d: @import("driver.zig").Driver = .{ .lib = undefined, .api = undefined, .write_value32 = write };
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemHostGetDevicePointer_v2 = devicePointer;
        d.api.cuMemFreeHost = freeHost;
        d.api.cuStreamSynchronize = sync;
        d.api.cuStreamQuery = query;
        return d;
    }
    /// Sets the process mode for one test; false when the test needs this host's WFET and it has none.
    fn use(m: Mode) bool {
        reset();
        if (m == .on and !(arm and @as(u64, auxval(std.elf.AT.HWCAP2)) & hwcap2_wfxt != 0)) return false;
        mode.store(m, .release);
        return true;
    }
};
test "host wait: the setting is on only when asked on a host with WFET" {
    try std.testing.expectEqual(Mode.off, modeOf(false, hwcap2_wfxt));
    try std.testing.expectEqual(Mode.no_wfxt, modeOf(true, 0));
    try std.testing.expectEqual(if (arm) Mode.on else Mode.no_wfxt, modeOf(true, hwcap2_wfxt));
}
test "host wait: unset, no marker is queued and the wait is the existing synchronize" {
    _ = Fake.use(.off);
    defer mode.store(.unread, .release);
    const d = Fake.driver();
    var w: Waiter = .{};
    try w.synchronize(Stream.borrowed(&d, null));
    try std.testing.expect(Fake.writes == 0 and Fake.syncs == 1 and w.buf == null);
    w.deinit();
}
test "host wait: set without HWCAP2_WFXT, no marker is queued and the wait is the existing synchronize" {
    _ = Fake.use(modeOf(true, 0));
    defer mode.store(.unread, .release);
    const d = Fake.driver();
    var w: Waiter = .{};
    try w.synchronize(Stream.borrowed(&d, null));
    try std.testing.expect(Fake.writes == 0 and Fake.syncs == 1 and w.buf == null);
    w.deinit();
}
test "host wait: a marker the GPU stored ends the wait, and teardown frees its word" {
    if (!Fake.use(.on)) return error.SkipZigTest;
    defer mode.store(.unread, .release);
    const d = Fake.driver();
    var w: Waiter = .{};
    try w.synchronize(Stream.borrowed(&d, null));
    try w.synchronize(Stream.borrowed(&d, null));
    try std.testing.expect(!w.refused and w.pending == null and w.seq == 2 and Fake.writes == 2 and Fake.syncs == 2);
    w.deinit();
    try std.testing.expectEqual(@as(usize, 1), Fake.frees);
}
test "host wait: a wait whose marker never lands ends at its deadlines on the stream query" {
    if (!Fake.use(.on)) return error.SkipZigTest;
    defer mode.store(.unread, .release);
    const d = Fake.driver();
    var w: Waiter = .{};
    Fake.gpu_writes = false;
    Fake.query_results = &.{ abi.error_not_ready, abi.error_not_ready };
    try w.synchronize(Stream.borrowed(&d, null));
    try std.testing.expect(!w.refused and w.pending == null and Fake.queries == 3 and Fake.syncs == 1);
    w.deinit();
}
test "host wait: a marker error falls back to the existing synchronize, and the waiter stays on it" {
    if (!Fake.use(.on)) return error.SkipZigTest;
    defer mode.store(.unread, .release);
    const d = Fake.driver();
    var w: Waiter = .{};
    Fake.write_result = 1; // CUDA_ERROR_INVALID_VALUE
    try w.synchronize(Stream.borrowed(&d, null));
    try std.testing.expect(w.refused and w.pending == null and Fake.syncs == 1);
    try w.synchronize(Stream.borrowed(&d, null));
    try std.testing.expect(Fake.writes == 1 and Fake.syncs == 2);
    w.deinit();
}
test "host wait: a stream query error falls back to the existing synchronize" {
    if (!Fake.use(.on)) return error.SkipZigTest;
    defer mode.store(.unread, .release);
    const d = Fake.driver();
    var w: Waiter = .{};
    Fake.gpu_writes = false;
    Fake.query_results = &.{999};
    try w.synchronize(Stream.borrowed(&d, null));
    try std.testing.expect(w.refused and w.pending == null and Fake.queries == 1 and Fake.syncs == 1);
    w.deinit();
}
test "host wait: a failed synchronize after a queued marker keeps the word at teardown" {
    if (!Fake.use(.on)) return error.SkipZigTest;
    defer mode.store(.unread, .release);
    const d = Fake.driver();
    var w: Waiter = .{};
    Fake.gpu_writes = false;
    Fake.sync_result = abi.error_not_ready;
    try std.testing.expectError(error.NotReady, w.synchronize(Stream.borrowed(&d, null)));
    try std.testing.expect(w.pending != null);
    w.deinit();
    try std.testing.expect(Fake.frees == 0 and w.buf == null and w.pending == null);
}
test "host wait: park returns when the word already differs, and within its slice when it does not" {
    var word: u32 = 8;
    park(&word, 7);
    if (!(arm and @as(u64, auxval(std.elf.AT.HWCAP2)) & hwcap2_wfxt != 0)) return;
    word = 7;
    park(&word, 7);
}
