The authors gratefully acknowledge Jonas Lemke for the development of the original experimental PPO and measurement-control framework (https://github.com/Jonas-Lemke/afc_with_ppo) on which substantial parts of the present implementation are based. His Master’s thesis at TU Berlin [2025] also served as a motivation to use a finite reward horizon for temporal credit assignment.

# afc_with_ppo

## Table of Contents
- [afc\_with\_ppo](#afc_with_ppo)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
    - [NI-DAQmx Driver Installation](#ni-daqmx-driver-installation)

## Prerequisites

Before running this project, certain drivers and a Python environment need to be set up.

### NI-DAQmx Driver Installation

This project requires NI-DAQmx drivers to interface with National Instruments data acquisition hardware.

**General Steps to Acquire NI-DAQmx Drivers:**

**Visit the National Instruments Website:** The official source for NI drivers is the National Instruments website. You can typically start your search at the [NI-DAQmx Driver](https://www.ni.com/de/support/downloads/drivers/download.ni-daq-mx.html#565026). If the link is outdated, download it manually:

1.  **Search for Downloads:** On the NI website, look for the "Downloads" or "Drivers" section [NI Downloads page](https://www.ni.com/downloads/). Search directly for "NI-DAQmx driver".
2.  **Download and Install:** Follow the installation instructions provided by National Instruments for your specific operating system.

**Supported NI-DAQmx Driver Versions:**

Some functions in the nidaqmx package may be unavailable with earlier versions of the NI-DAQmx driver. Refer to the Installation section for details on how to install the latest version of the NI-DAQmx driver.

**Operating System Support:**

The NI-DAQmx driver is for **Windows** systemy only supported (NI USB-6281). Refer to NI Hardware and Operating System Compatibility for which versions of the driver support your hardware on a given operating system.
