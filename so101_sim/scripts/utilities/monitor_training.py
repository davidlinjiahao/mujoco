#!/usr/bin/env python3
"""
Monitor training progress with a live progress bar
"""

import time
import re
from pathlib import Path
from datetime import datetime, timedelta

log_file = Path("training_overnight.log")
total_steps = 200000

print("=" * 80)
print("🤖 SO-101 TRAINING MONITOR")
print("=" * 80)
print(f"Target: {total_steps:,} steps")
print(f"Log file: {log_file}")
print("=" * 80)
print()

last_step = 0
start_time = None
steps_per_second = None

while True:
    try:
        if not log_file.exists():
            print("⏳ Waiting for training to start...")
            time.sleep(5)
            continue
        
        # Read last line with step info
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Find the most recent step line
        for line in reversed(lines):
            if 'step:' in line:
                match = re.search(r'step:(\d+)', line)
                if match:
                    current_step = int(match.group(1))
                    
                    # Parse loss if available
                    loss_match = re.search(r'loss:([\d.]+)', line)
                    loss = float(loss_match.group(1)) if loss_match else None
                    
                    # Calculate speed
                    if start_time is None:
                        start_time = datetime.now()
                    else:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        if elapsed > 0:
                            steps_per_second = current_step / elapsed
                    
                    # Only update if step changed
                    if current_step != last_step:
                        last_step = current_step
                        
                        # Calculate progress
                        progress = (current_step / total_steps) * 100
                        
                        # Progress bar
                        bar_length = 50
                        filled = int(bar_length * current_step / total_steps)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        
                        # Time estimates
                        if steps_per_second and steps_per_second > 0:
                            remaining_steps = total_steps - current_step
                            eta_seconds = remaining_steps / steps_per_second
                            eta = timedelta(seconds=int(eta_seconds))
                            speed_str = f"{steps_per_second:.2f} steps/s"
                            eta_str = f"ETA: {eta}"
                        else:
                            speed_str = "calculating..."
                            eta_str = "ETA: calculating..."
                        
                        # Clear and print status
                        print(f"\r\033[K", end='')  # Clear line
                        print(f"Step {current_step:,}/{total_steps:,} [{bar}] {progress:.1f}%", end='')
                        print(f" | {speed_str} | {eta_str}", end='')
                        
                        if loss:
                            print(f" | Loss: {loss:.3f}", end='')
                        
                        # Check for checkpoints at milestones
                        if current_step % 10000 == 0 and current_step > 0:
                            print(f"\n✅ Checkpoint saved at {current_step:,} steps!")
                        
                        # Special alerts
                        if current_step == 50000:
                            print(f"\n🎯 MILESTONE: 50K steps reached! (25% complete)")
                        elif current_step == 100000:
                            print(f"\n🎯 MILESTONE: 100K steps reached! (50% complete) - Ready to evaluate!")
                        elif current_step == 150000:
                            print(f"\n🎯 MILESTONE: 150K steps reached! (75% complete)")
                    
                    break
        
        time.sleep(5)  # Update every 5 seconds
        
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped.")
        print(f"📊 Final step: {last_step:,}/{total_steps:,}")
        break
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        time.sleep(5)

