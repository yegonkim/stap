#!/bin/bash

# Start a new tmux session
tmux new-session -d -s $S

# Split the window into panes
tmux split-window -v
tmux split-window -h
tmux split-window -h
tmux select-pane -t 0
tmux split-window -h

# Attach to the session
tmux attach-session -t $S
