# gr-echo
Repo holding out-of-tree gnuradio module for Echo protocol modulation learning

## Reed-Solomon Encoder-Decoder
```sh
git submodule update --init # Clone library
cd misc/
./apply-patches.sh # Apply patch to reedsolomon, build, and install
```

Needed for the internal packet info - sequence number and invalid feedback flag.

## Packet Structure
| BER Spy Hdr | Pkt Info Hdr | Preamble | Echo | 
|:---:|:---:|:---:|:---:|
| 64 symbols  | 96 **bits** | 256 symbols | 256 symbols |

## Patches
### GNURadio
`misc/zmq-bind.patch` adds a bind/connect flag to the zmq blocks so that we can connect to them from the outside to send commands. Requires GNURadio `v3.7.9`.

Copy it to the top level of the gnr source, apply it, and rebuild _BEFORE_ you open any of the example flowgraphs for the first time. If you open a flowgraph and save it before this patch is applied, the settings for appropriate bind/connect behavior in the control ports will be lost.

### reedsolomon
`misc/reedsolomon.patch` fixes a broken implicit `str` decode-encode.
