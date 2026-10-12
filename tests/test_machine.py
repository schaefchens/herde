"""What a worker tells its projects about its computer: an AMD GPU, found where Linux lists its GPUs."""

from herde.machine import amd_gpu

# libdrm's amdgpu.ids: its head, then lines from the list ROCm's PyTorch brings along, and from Debian 12's.
HEAD = """\
# List of AMDGPU IDs
#
# Syntax:
# device_id,\trevision_id,\tproduct_name        <-- single tab after comma

1.0.0
"""
NEWER = HEAD + "1586,\tC1,\tAMD Radeon 8060S Graphics\n744C,\tC8,\tAMD Radeon RX 7900 XTX\n744C,\tCC,\tAMD Radeon RX 7900 XT\n"
OLDER = HEAD + "73BF,\tC1,\tAMD Radeon RX 6800 XT\n"


def card(drm, name, vendor="0x1002", device="0x1586", revision="0xc1", vram=None):
    """A card as /sys/class/drm shows it: its PCI device's ids, and for AMD the memory of its own."""
    dev = drm / name / "device"
    dev.mkdir(parents=True)
    for file, value in (("vendor", vendor), ("device", device), ("revision", revision)):
        (dev / file).write_text(value + "\n")
    if vram is not None:
        (dev / "mem_info_vram_total").write_text(f"{vram}\n")


def lists(tmp_path):
    (tmp_path / "newer.ids").write_text(NEWER)
    (tmp_path / "older.ids").write_text(OLDER)
    return tmp_path / "newer.ids", tmp_path / "older.ids"


def test_the_amd_gpu_with_the_most_memory_of_its_own_by_amds_name(tmp_path):
    drm, ids = tmp_path / "drm", lists(tmp_path)
    card(drm, "card0", device="0x1586", revision="0xc1", vram=512 * 2**20)
    assert amd_gpu(drm, ids) == "AMD Radeon 8060S Graphics", "a Ryzen AI Max alone: its APU"
    card(drm, "card1", device="0x744c", revision="0xc8", vram=24 * 2**30)
    card(drm, "card1-DP-1", device="0x744c", revision="0xcc", vram=48 * 2**30)
    card(drm, "card2", vendor="0x10de", device="0x2684", revision="0xa1", vram=96 * 2**30)
    assert amd_gpu(drm, ids) == "AMD Radeon RX 7900 XTX", "the card before the APU; a connector and another vendor's GPU are no AMD GPU"


def test_every_list_is_asked_and_without_one_the_pci_device_is_named(tmp_path):
    drm = tmp_path / "drm"
    card(drm, "card0", vendor="0x8086", device="0xa780", revision="0x04")
    assert amd_gpu(drm, lists(tmp_path)) == ""
    card(drm, "card1", device="0x73bf", revision="0xc1")
    assert amd_gpu(drm, lists(tmp_path)) == "AMD Radeon RX 6800 XT", "the newer list does not have it; the older one does"
    assert amd_gpu(drm, (tmp_path / "none.ids",)) == "AMD GPU 73BF"
    assert amd_gpu(tmp_path / "no-drm") == "", "no /sys/class/drm (a Mac)"
