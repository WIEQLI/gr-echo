# Common variables and functions for run scripts

# You should replace these fields as needed for your own USRP setup.
# Our USRPs are accessed via LXD containers on servers connected to 
# the USRPs, hence the ssh indirection.
SRN1="<user>@usrp-001.<domain>"
SRN2="<user>@usrp-002.<domain>"
RADIO1="192.168.42.233"
RADIO2="192.168.42.95"

SEED_MOD="$(od -v -An -N4 -tu4 < /dev/urandom | tr -cd '[:alnum:]')"
SEED_DEMOD="$(od -v -An -N4 -tu4 < /dev/urandom | tr -cd '[:alnum:]')"
SEED_MOD_2="$(od -v -An -N4 -tu4 < /dev/urandom | tr -cd '[:alnum:]')"
SEED_DEMOD_2="$(od -v -An -N4 -tu4 < /dev/urandom | tr -cd '[:alnum:]')"

RUN_DIR="/root/$1-$2-${SEED_MOD}"
txgain="$3"
rxgain="$4"
bps="$5"
maxamp="$6"
demod_init="$7"
mod_init="$8"
shared_preamble="$9"

if [[ x != "x${demod_init}" || x != "x${mod_init}" ]]; then
    RUN_DIR="/root/pretrained-$1-$2-${SEED_MOD}"
fi
if [[ x != "x${shared_preamble}" ]]; then
    RUN_DIR="/root/shared-$1-$2-${SEED_MOD}"
fi
echo "Run directory: ${RUN_DIR}"

CLASSIC_CMD="mkdir -p ${RUN_DIR}; cd ${RUN_DIR}; screen -c /root/gr-echo/misc/screenrc -d -m /root/gr-echo/examples/echo_single_usrp.py --bits-per-symb $bps --modtype=classic --demodtype=classic --tx-gain $txgain --rx-gain $rxgain --max-amplitude $maxamp --shared-preamble \"$shared_preamble\" --lambda-center 125 --log-interval=50"
NEURAL_CMD="mkdir -p ${RUN_DIR}; cd ${RUN_DIR}; screen -c /root/gr-echo/misc/screenrc -d -m /root/gr-echo/examples/echo_single_usrp.py --bits-per-symb $bps --modtype=neural --demodtype=neural --mod-seed SEED_MOD --demod-seed SEED_DEMOD --tx-gain $txgain --rx-gain $rxgain --max-amplitude $maxamp --demod-init-weights \"$demod_init\" --mod-init-weights \"$mod_init\" --shared-preamble \"$shared_preamble\" --lambda-center 125 --log-interval=50"
NEURAL_CMD_2="${NEURAL_CMD/SEED_MOD/$SEED_MOD_2}"
NEURAL_CMD_2="${NEURAL_CMD_2/SEED_DEMOD/$SEED_DEMOD_2}"
NEURAL_CMD="${NEURAL_CMD/SEED_MOD/$SEED_MOD}"
NEURAL_CMD="${NEURAL_CMD/SEED_DEMOD/$SEED_DEMOD}"
KILL_CMD="screen -X quit"
REFLASH_CMD="/root/reflash-me.sh"
FREEZE_CMD="/root/gr-echo/misc/echo-ctrl.py -f"
SPY_CMD="mkdir -p ${RUN_DIR}; cp /root/spy_master.npy ${RUN_DIR}/"

run_radio() {
    _srn="$1"
    _radio_ip="$2"
    _cmd="$3"

    ssh "${_srn}" -C "ssh root@${_radio_ip} -C '${_cmd}'"
}

