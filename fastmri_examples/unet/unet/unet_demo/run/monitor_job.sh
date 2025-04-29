#!/bin/bash

# === CONFIGURATION ===
EMAIL="leona.j.burk@gmail.com"
SUBJECT="[FastMRI] Multi-Job Training Update"
TMP_FILE="/tmp/job_monitor_$$.txt"

# Job IDs to monitor
JOB_IDS=(2526791 2527421 2527445)

# Map job IDs to lightning log paths
# Update the paths accordingly per job ID
declare -A LOGDIRS=(
    [2526791]="/home/hice1/lburk3/scratch/fastMRI/fastmri_examples/unet/unet/unet_demo/lightning_logs/version_2526791"
    [2527421]="/home/hice1/lburk3/scratch/fastMRI/fastmri_examples/unet/unet/unet_demo/lightning_logs/version_2527421"
    [2527445]="/home/hice1/lburk3/scratch/fastMRI/fastmri_examples/unet/unet/unet_demo/lightning_logs/version_2527445"
)

# === MONITOR LOOP ===
while true; do
    {
        echo "FastMRI Multi-Job Training Update ($(date))"
        echo "--------------------------------------------"

        for JOB_ID in "${JOB_IDS[@]}"; do
            echo "== Job ID: $JOB_ID =="

            # Get job status
            JOB_STATUS=$(sacct -j $JOB_ID --format=State --noheader | head -n 1 | awk '{print $1}')

            # Get start time and elapsed
            START_TIME_RAW=$(sacct -j $JOB_ID --format=Start --noheader | head -n 1)
            ELAPSED_RAW=$(sacct -j $JOB_ID --format=Elapsed --noheader | head -n 1)

            if [[ "$JOB_STATUS" == RUNNING ]]; then
                echo "Status           : RUNNING"
                START_LINE="Running on $(hostname) with $(nproc) CPUs"
                echo "Started          : ${START_LINE}"
                echo "Elapsed Time     : ${ELAPSED_RAW:-N/A}"

                LOGDIR="${LOGDIRS[$JOB_ID]}"
                EVENT_FILE=$(ls -t "$LOGDIR"/events.out.tfevents.* 2>/dev/null | head -n 1)

                if [ -f "$EVENT_FILE" ]; then
                    EPOCH=$(python3 - <<END
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
try:
    ea = EventAccumulator("$LOGDIR")
    ea.Reload()
    scalars = ea.Scalars("epoch")
    print(int(scalars[-1].value) if scalars else "N/A")
except:
    print("N/A")
END
)

                    VAL_LOSS=$(python3 - <<END
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
try:
    ea = EventAccumulator("$LOGDIR")
    ea.Reload()
    scalars = ea.Scalars("validation_loss")
    print(round(scalars[-1].value, 6) if scalars else "N/A")
except:
    print("N/A")
END
)

                    echo "Latest Epoch     : $EPOCH"
                    echo "Validation Loss  : $VAL_LOSS"
                else
                    echo "Latest Epoch     : N/A"
                    echo "Validation Loss  : N/A"
                fi
            else
                echo "Status           : ${JOB_STATUS:-Unknown}"
                echo "Started          : Not yet started"
                echo "Elapsed Time     : N/A"
            fi

            echo ""
        done
    } > "$TMP_FILE"

    # Send the email
    mail -s "$SUBJECT" "$EMAIL" < "$TMP_FILE"

    # Wait 15 minutes
    sleep 900
done