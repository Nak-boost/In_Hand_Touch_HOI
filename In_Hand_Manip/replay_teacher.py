"""Compatibility entry point; use replay_policy.py for teacher or student replay."""
from replay_policy import main

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
