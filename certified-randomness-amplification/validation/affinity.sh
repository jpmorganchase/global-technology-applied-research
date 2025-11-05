#!/usr/bin/env bash

# Get the RankID from different launcher
if [[ -v MPI_LOCALRANKID ]]; then
  _MPI_RANKID=$MPI_LOCALRANKID 
elif [[ -v PALS_LOCAL_RANKID ]]; then
  _MPI_RANKID=$PALS_LOCAL_RANKID
else
  echo Error
  exit 1
fi

#This give the exact GPU count i915 knows about and I use udev to only enumerate the devices with physical presence.
num_gpu=$(/usr/bin/udevadm info /sys/module/i915/drivers/pci:i915/* |& grep -v Unknown | grep -c "P: /devices")
num_tile=2
gpu_id=$(((_MPI_RANKID / num_tile) % num_gpu))
tile_id=$((_MPI_RANKID % num_tile))

unset EnableWalkerPartition
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
if [ "$_MPI_RANKID" -eq 12 ]; then
  export ZE_AFFINITY_MASK=0
elif [ "$_MPI_RANKID" -eq 13 ]; then
  export ZE_AFFINITY_MASK=1
else
    export ZE_AFFINITY_MASK=$_MPI_RANKID # The frameworks module set ZE_FLAT_DEVICE_HIERARCHY=FLAT, whereas gpu_tile_compact.sh requires it to be COMPOSITE. Use gpu_dev_compact.sh instead.
fi
ulimit -c 0 # Until Aurora filesystem problems are fixed

# echo $_MPI_RANKID, $ZE_AFFINITY_MASK
# echo "gpu_id: $gpu_id, tile_id: $tile_id"

python -u main_torch.py
#https://stackoverflow.com/a/28099707/7674852
# "$@"
