"""
S2S Physics Audit Visualizer — matplotlib plots for certification results.

Shows acceleration magnitude with certification tier highlighted.
Requires: pip install matplotlib numpy

Usage:
    from s2s_standard_v1_3.visualizer import plot_certification, plot_session
"""

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    _PLOT = True
except ImportError:
    _PLOT = False

TIER_COLORS = {
    'GOLD':     '#FFD700',
    'SILVER':   '#C0C0C0',
    'BRONZE':   '#CD7F32',
    'REJECTED': '#FF4444',
}

def plot_certification(imu_raw: dict, result: dict,
                       title: str = "S2S Physics Audit",
                       save_path: str = None) -> None:
    """
    Plot acceleration magnitude with certification tier overlay.

    Args:
        imu_raw: dict with 'timestamps_ns' and 'accel'
        result:  certify() output dict
        title:   plot title
        save_path: if set, saves to file instead of showing
    """
    if not _PLOT:
        print("Install matplotlib: pip install matplotlib")
        return

    ts  = np.array(imu_raw['timestamps_ns']) * 1e-9
    ts -= ts[0]
    acc = np.array(imu_raw['accel'])
    mag = np.sqrt(np.sum(acc**2, axis=1))

    tier  = result.get('tier', 'UNKNOWN')
    score = result.get('physical_law_score', 0)
    color = TIER_COLORS.get(tier, '#888888')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"{title}\nTier: {tier} | Score: {score}/100", fontsize=13)

    # Acceleration magnitude
    ax1.plot(ts, mag, color='#333333', linewidth=0.8, label='|acceleration|')
    ax1.axhspan(ts[0], ts[-1], alpha=0.0)
    ax1.fill_between(ts, mag, alpha=0.15, color=color)
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Per-axis
    labels = ['X', 'Y', 'Z']
    colors = ['#E74C3C', '#2ECC71', '#3498DB']
    for i in range(min(3, acc.shape[1])):
        ax2.plot(ts, acc[:, i], color=colors[i],
                 linewidth=0.7, alpha=0.8, label=f'acc_{labels[i]}')
    ax2.set_ylabel('m/s²')
    ax2.set_xlabel('Time (s)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # Tier badge
    patch = mpatches.Patch(color=color, label=f'{tier} ({score}/100)')
    ax1.legend(handles=[patch,
               mpatches.Patch(color='#333333', label='|accel|')],
               loc='upper right')

    # Laws summary
    passed = result.get('laws_passed', [])
    failed = result.get('laws_failed', [])
    summary = f"PASS {len(passed)} passed  FAIL {len(failed)} failed"
    if failed:
        summary += f"\nFailed: {', '.join(failed)}"
    ax1.text(0.01, 0.95, summary, transform=ax1.transAxes,
             fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_session(timeline: list, session_id: str = "session",
                 save_path: str = None) -> None:
    """
    Plot session quality timeline — shows tier over time.

    Args:
        timeline: list of dicts from session_report()['quality_timeline']
        session_id: label for the plot
        save_path: if set, saves to file
    """
    if not _PLOT:
        print("Install matplotlib: pip install matplotlib")
        return

    if not timeline:
        print("No timeline data")
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle(f"S2S Session Quality — {session_id}", fontsize=13)

    tier_order = {'GOLD': 4, 'SILVER': 3, 'BRONZE': 2, 'REJECTED': 1}

    for seg in timeline:
        t_start = seg['start_s']
        t_end   = seg['end_s']
        tier    = seg['tier']
        color   = TIER_COLORS.get(tier, '#888888')
        score   = seg['mean_score']
        level   = tier_order.get(tier, 0)

        ax.barh(y=0, width=t_end - t_start, left=t_start,
                height=0.8, color=color, alpha=0.8, edgecolor='white')
        if t_end - t_start > 1.0:
            ax.text((t_start + t_end) / 2, 0,
                    f"{tier}\n{score:.0f}",
                    ha='center', va='center', fontsize=7, color='black')

    ax.set_xlabel('Time (s)')
    ax.set_yticks([])
    ax.set_xlim(0, max(s['end_s'] for s in timeline))

    legend = [mpatches.Patch(color=c, label=t)
              for t, c in TIER_COLORS.items()]
    ax.legend(handles=legend, loc='upper right', fontsize=9)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()
