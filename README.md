# Catena (working name)

Cloud masking conditioned on the imaging chain, for any sensor and any processor.

Give the model a satellite's spec sheet (bands, GSD, MTF, SNR, bit depth, processing chain)
and get a cloud mask on day zero, without labelled images from that satellite.

The code is built one file at a time; each step has a test that measures what the code does
and a demo that shows it.

## Build log

| step | file | what it does |
|---|---|---|
| A1 | `src/catena/sensors/physics.py` | MTF, noise and digitisation: how a sensor turns a scene into numbers |

## Running

    python -m venv .venv && . .venv/bin/activate
    pip install -e ".[dev]"
    pytest                                  # CPU only, a few seconds
    python examples/a1_imaging_chain.py     # writes figures/a1_*.png

Long jobs on the GPU machine run inside tmux, so they survive a dropped connection:

    tmux new -s catena          # start a session and run the job inside it
                                # Ctrl-b then d detaches; the job keeps running
    tmux attach -t catena       # come back later
