"""
Analyze gradient norms from TensorBoard event files
"""
import os
import sys

# Try to import tensorboard
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("tensorboard not installed, trying alternative method...")
    # Fallback: parse the log file
    log_file = "logs/44k/train.log"
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print("=" * 80)
        print("Gradient Analysis from Training Log")
        print("=" * 80)
        print(f"\nTotal log entries: {len(lines)}")
        print("\nNote: grad_norm values are NOT in the text log.")
        print("They are only written to TensorBoard events file.")
        print("\nTo view gradient data, please:")
        print("1. Check TensorBoard at http://localhost:6006")
        print("2. Look for 'grad_norm_d' and 'grad_norm_g' scalars")
        print("\nAlternatively, install tensorboard:")
        print("   pip install tensorboard")
        sys.exit(1)

def analyze_tensorboard_events(event_file):
    """Extract gradient norms from TensorBoard event file"""
    print(f"Loading event file: {event_file}")
    
    # Load the event file
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={
            event_accumulator.SCALARS: 0,  # 0 means load all
        }
    )
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()['scalars']
    print(f"\nAvailable scalar tags ({len(tags)} total):")
    for tag in sorted(tags):
        print(f"  - {tag}")
    
    # Extract gradient norms
    grad_tags = [tag for tag in tags if 'grad_norm' in tag]
    if not grad_tags:
        print("\n⚠️  No gradient norm data found in event file!")
        return
    
    print(f"\n{'='*80}")
    print("Gradient Norm Analysis")
    print(f"{'='*80}")
    
    for tag in grad_tags:
        events = ea.Scalars(tag)
        print(f"\n{tag}:")
        print(f"  Total data points: {len(events)}")
        
        if events:
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            print(f"  Step range: {min(steps)} - {max(steps)}")
            print(f"  Value range: {min(values):.4f} - {max(values):.4f}")
            print(f"  Average: {sum(values)/len(values):.4f}")
            
            # Find peaks (values > 10)
            peaks = [(s, v) for s, v in zip(steps, values) if v > 10]
            if peaks:
                print(f"\n  ⚠️  WARNING: Found {len(peaks)} steps with grad_norm > 10:")
                for step, value in peaks[:10]:  # Show first 10
                    print(f"    Step {step}: {value:.4f}")
                if len(peaks) > 10:
                    print(f"    ... and {len(peaks) - 10} more")
            
            # Check for specific problematic steps
            problematic_steps = [2600, 2800]
            for target_step in problematic_steps:
                nearby = [(s, v) for s, v in zip(steps, values) 
                         if abs(s - target_step) <= 50]
                if nearby:
                    print(f"\n  Around step {target_step}:")
                    for step, value in nearby:
                        marker = " ⚠️ HIGH" if value > 10 else ""
                        print(f"    Step {step}: {value:.4f}{marker}")

if __name__ == "__main__":
    event_file = "logs/44k/events.out.tfevents.1776044563.mahiro.10216.0"
    
    if not os.path.exists(event_file):
        print(f"Error: Event file not found: {event_file}")
        sys.exit(1)
    
    analyze_tensorboard_events(event_file)
