#!/bin/bash
set -e

GETOPT=getopt
if [ -e /usr/local/bin/gnu-getopt ]; then
    GETOPT=gnu-getopt
fi

args_list=(
    "bps:"
    "cc"
    "demod-init-weights:"
    "help"
    "max-amplitude:"
    "mod-init-weights:"
    "nc"
    "nn"
    "preamble:"
    "rx-gain:"
    "sync"
    "tx-gain:"
    "time:"
)

opts=$($GETOPT \
    --longoptions "$(printf "%s," "${args_list[@]}")" \
    --name "$(basename "$0")" \
    --options "a:b:hp:st:" \
    -- "$@"
)

# Setup defaults
mode1=classic
mode2=classic
twait=30
sync=0
txgain=22
rxgain=23
bps=2
maxamp="1.0"
demod_init=""
mod_init=""
preamble=""

eval set -- "$opts"
echo $@
while [[ $# -gt 0 ]]; do
    case "$1" in
        -a | --max-amplitude)
            maxamp="$2"
            shift 2
            ;;
        -b | --bps)
            bps=$2
            shift 2
            ;;
        --cc)
            mode1=classic
            mode2=classic
            shift
            ;;
        --demod-init-weights)
            demod_init="$2"
            shift 2
            ;;
        --mod-init-weights)
            mod_init="$2"
            shift 2
            ;;
        --nc)
            mode1=neural
            mode2=classic
            shift
            ;;
        --nn)
            mode1=neural
            mode2=neural
            shift
            ;;
        -p | --preamble)
            preamble="$2"
            shift 2
            ;;
        --rx-gain)
            rxgain="$2"
            shift 2
            ;;
        --tx-gain)
            txgain="$2"
            shift 2
            ;;
        -t | --time)
            twait="$2"
            shift 2
            ;;
        -s | --sync)
            sync=1 
            shift
            ;;
        -h | --help)
            echo "Usage: $(basename $0) {--cc,--nc,--nn} [-b/--bps BPS] [--rx-gain GAIN] [--tx-gain GAIN] [-a/--max-amplitude MAX_AMP] [-t/--time TIME] [--demod-init-weights WGTS] [--mod-init-weights WGTS] [-p/--preamble SHARED_PREAMBLE] [-s/--sync] [-h/--help]"
            exit 0
            ;;
        --) 
            shift
            ;;
        *)
            echo "Unhandled arg $1"
            exit 1
            ;;
    esac
done

echo "Mode is $mode1-$mode2 for duration $twait"

. common.sh $mode1 $mode2 $txgain $rxgain $bps $maxamp "$demod_init" "$mod_init" "$preamble"

if [[ $mode1 == "classic" ]]; then CMD1="$CLASSIC_CMD"; else CMD1="$NEURAL_CMD"; fi
if [[ $mode2 == "classic" ]]; then CMD2="$CLASSIC_CMD"; else CMD2="$NEURAL_CMD_2"; fi
echo "CMD1 $CMD1"
echo "CMD2 $CMD2"

# Copy spy master files
echo "Copying spy master on srn1"
run_radio $SRN1 $RADIO1 "$SPY_CMD"
echo "Copying spy master on srn2"
run_radio $SRN2 $RADIO2 "$SPY_CMD"

# Start flowgraph on srn1
echo "Starting radio 1"
run_radio $SRN1 $RADIO1 "$CMD1"

# Start flowgraph on srn2
echo "Starting radio 2"
run_radio $SRN2 $RADIO2 "$CMD2"

# Wait for 30 seconds
echo "Waiting for startup..."
sleep 12
if [[ $mode1 == "classic" && $mode2 == "classic" ]]; then
    echo "Both agents are classic, freezing for BER estimation"
    run_radio $SRN1 $RADIO1 "$FREEZE_CMD"
    run_radio $SRN2 $RADIO2 "$FREEZE_CMD"
fi
echo "Waiting for run..."
sleep "$twait"

if [[ $mode1 != "classic" || $mode2 != "classic" ]]; then
    # Freeze training
    echo "Freezing training..."
    run_radio $SRN1 $RADIO1 "$FREEZE_CMD"
    run_radio $SRN2 $RADIO2 "$FREEZE_CMD"

    # Wait for 60 seconds to get a BER estimate
    echo "Collecting BER estimates..."
    sleep 60
else
    echo "Both agents are classic, BER estimation complete"
fi

# Stop classic on srn1
echo "Stopping radio 1"
run_radio $SRN1 $RADIO1 "$KILL_CMD"

# Stop classic on srn2
echo "Stopping radio 2"
run_radio $SRN2 $RADIO2 "$KILL_CMD"

# Reflash USRPs
echo "Reflashing USRPs"
run_radio $SRN1 $RADIO1 "$REFLASH_CMD"
run_radio $SRN2 $RADIO2 "$REFLASH_CMD"

# Copy down files
if [[ $sync -eq 1 ]]; then
    echo "Copying files from SRN1"
    srn-rsync.sh 1 "$RUN_DIR"
    mv $(basename "$RUN_DIR") "$(basename "$RUN_DIR")-srn1"
    
    echo "Copying files from SRN2"
    srn-rsync.sh 2 "$RUN_DIR"
    mv $(basename "$RUN_DIR") "$(basename "$RUN_DIR")-srn2"
fi

