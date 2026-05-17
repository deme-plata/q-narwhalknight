# #038: OsTuner hardcodes device names

**Priority**: LOW
**File(s)**: `crates/q-compute/src/os_tuner.rs`
**Risk**: I/O scheduler tuning silently skipped on non-standard hardware

## Problem

`tune_io_scheduler()` (line 107) hardcodes a fixed list of block device names:

```rust
let devices = ["sda", "nvme0n1", "vda"];
```

This only covers:
- `sda` — first SATA/SCSI disk
- `nvme0n1` — first NVMe drive
- `vda` — first virtio disk (KVM/QEMU)

It misses:
- `sdb`, `sdc`, etc. — multi-disk systems
- `nvme1n1`, `nvme2n1`, etc. — multi-NVMe systems (common on compute nodes)
- `xvda` — Xen virtual disks (AWS older instances)
- `mmcblk0` — eMMC/SD card based systems (ARM SBCs)
- Any RAID controller device names

On Epsilon (10Gbit supernode), which has multiple NVMe drives, only the first drive gets its I/O scheduler tuned.

## Fix

1. Enumerate actual block devices by reading `/sys/block/` directory entries instead of using a hardcoded list:
   ```rust
   for entry in std::fs::read_dir("/sys/block/").ok()? {
       let name = entry.ok()?.file_name().to_string_lossy().to_string();
       if name.starts_with("loop") || name.starts_with("dm-") || name.starts_with("ram") {
           continue;
       }
       // Apply appropriate scheduler based on device type
   }
   ```
2. Detect whether a device is rotational via `/sys/block/{dev}/queue/rotational` (0 = SSD/NVMe, 1 = HDD) instead of pattern-matching names.
3. Log which devices were tuned and which were skipped for operator visibility.

## Testing

- cargo check --package q-compute
- cargo test --package q-compute
