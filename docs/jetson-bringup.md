# Jetson Orin Nano bring-up (S0.1, prerequisite)

Bringing a factory-fresh Orin Nano Developer Kit to the point where
`scripts/jetson-setup-check.sh` can run. Written against NVIDIA's current
documentation (checked 2026-09-12); the install flow changed since PLAN.md
was drafted, so the steps below supersede §9's assumptions.

## What changed since the plan

PLAN.md assumed the old "download an SD card image, write it, boot" flow.
NVIDIA now ships a **Jetson ISO installer** (`r39.2.1`, 2026-08-07) that you
write to a *USB stick*; booting it installs Jetson Linux onto the microSD or
NVMe. Two consequences:

- You need a USB stick **and** the target microSD, not just the microSD.
- The image is an ISO: it must be written with balenaEtcher or `dd`, not
  copied in Finder.

## Hardware needed

| Item | Notes |
|---|---|
| USB stick, 8 GB+ | holds the installer ISO |
| microSD, 64 GB+ UHS-1 | the install target (an NVMe SSD also works) |
| Power supply | ships with the dev kit |
| **Display or serial** | see below — this is the easy thing to get stuck on |
| USB-C cable | only needed later, for SSH-over-USB from the Mac |

### The display gotcha

The Orin Nano Developer Kit outputs **DisplayPort, not HDMI**. A spare HDMI
monitor will not work without an active DP→HDMI adapter. Two options for
first boot:

1. **DisplayPort monitor** + USB keyboard/mouse.
2. **Headless over serial** — a 3.3 V USB-to-TTL cable (FTDI / CP2102) on the
   carrier board's button header. Then from the Mac:
   `screen /dev/tty.usbserial-* 115200`

Either gets you through first boot. After that the board is reachable over
USB-C at `192.168.55.1` and no display is needed again.

## Firmware: check before anything else

Dev kits shipped with factory QSPI firmware older than **36.0** cannot boot
JetPack 6.x or newer. This is a one-time update.

On a booted system:

```
sudo nvbootctrl dump-slots-info     # look for "Current version"
```

Chicken-and-egg on a fresh unit: if the modern installer boots and completes,
the firmware was already fine. If it refuses, take the microSD-only recovery
path — **no x86 Linux host or SDK Manager required**:

1. Write the JetPack 5.1.3 SD image (`JP513-orin-nano-sd-card-image_b29.zip`)
   to the microSD and boot it.
2. Complete first-boot setup, get it online, then:

   ```
   sudo apt update
   sudo apt install nvidia-l4t-jetson-orin-nano-qspi-updater
   ```

3. Reboot and let the QSPI update finish.
4. Power off, swap to the target install media, and run the modern installer.

Units manufactured recently often ship with 36.x already, so try the normal
path first rather than pre-emptively downgrading.

## Install

1. Download the installer ISO
   (`jetsoninstaller-r39.2.1-…-arm64.iso`) from NVIDIA.
2. Write it to the USB stick with [balenaEtcher](https://etcher.balena.io).
   Finder copy does not produce a bootable stick.
3. Insert the microSD into the slot on the **underside of the module**.
4. Plug in USB stick, display (or serial), keyboard, then power.
5. The board boots the installer; follow it to install onto the microSD.
6. Complete the Ubuntu first-boot wizard (user, locale, network).

## Then: enable the Mac workflow

Once booted, connect the Orin's USB-C to the Mac. It enumerates as a USB
network device:

```
ssh <user>@192.168.55.1
```

Copy the Mac's key over so the loop is passwordless, then clone the repo and
run the S0.1 check:

```
ssh-copy-id <user>@192.168.55.1
ssh <user>@192.168.55.1
git clone https://github.com/shaostassen/CUDA_MPPI_Controller.git
cd CUDA_MPPI_Controller
scripts/jetson-setup-check.sh          # writes docs/env.md from the real machine
```

If `nvcc` is missing from `PATH` (common — JetPack installs it but does not
always export it):

```
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

Then build and run everything that has been waiting on hardware:

```
cmake --preset release && cmake --build --preset release
scripts/gpu-test.sh                     # S2.2/S2.3/S2.4/S2.5 parity ACs
scripts/jetson-setup-check.sh           # re-run to capture device properties
```

## Sources

- [Quick Start Guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/quick_start.html)
- [JetPack 6.x Update Path / QSPI firmware](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/update_firmware.html)
- [JetPack SDK](https://developer.nvidia.com/embedded/jetpack)
