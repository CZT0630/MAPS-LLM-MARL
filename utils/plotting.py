# utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import os


class Plotter:
    """绘制训练和评估的指标"""
    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self._apply_default_style()

    def _apply_default_style(self):
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = (10, 5)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

    def _moving_average(self, data, window_size):
        if not data:
            return []
        window_size = max(1, int(window_size))
        out = []
        for i in range(len(data)):
            start = max(0, i - window_size + 1)
            out.append(np.mean(data[start:i+1]))
        return out

    def plot_rewards(self, rewards, ma_window=50):
        """绘制奖励曲线（含移动平均与统一风格）"""
        plt.figure()
        plt.plot(rewards, color='crimson', linestyle='-', marker='^', markersize=4, label='Reward')
        if isinstance(ma_window, int) and len(rewards) > ma_window:
            ma = self._moving_average(rewards, ma_window)
            plt.plot(ma, color='deepskyblue', linestyle='--', label=f'MA({ma_window})')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.legend()
        png_path = os.path.join(self.save_dir, 'rewards.png')
        pdf_path = os.path.join(self.save_dir, 'rewards.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_action_distribution(self, actions):
        """绘制动作分布"""
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(actions[:, 0], bins=20, alpha=0.7, color='goldenrod')
        plt.xlabel('Offload Ratio')
        plt.ylabel('Frequency')
        plt.title('Offload Ratio Distribution')
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.hist(actions[:, 1], bins=10, alpha=0.7, color='limegreen')
        plt.xlabel('Target Node')
        plt.ylabel('Frequency')
        plt.title('Target Node Distribution')
        plt.grid(True)
        plt.tight_layout()
        png_path = os.path.join(self.save_dir, 'action_distribution.png')
        pdf_path = os.path.join(self.save_dir, 'action_distribution.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_losses(self, metrics_tracker):
        """绘制训练损失曲线"""
        if not metrics_tracker.critic_losses or not metrics_tracker.actor_losses:
            return
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        plt.plot(metrics_tracker.critic_losses, color='deepskyblue')
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        plt.title('Critic Loss')
        plt.grid(True)
        plt.subplot(2, 2, 2)
        plt.plot(metrics_tracker.actor_losses, color='crimson')
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        plt.title('Actor Loss')
        plt.grid(True)
        plt.subplot(2, 2, 3)
        plt.plot(metrics_tracker.distillation_losses, color='purple')
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        plt.title('Distillation Loss')
        plt.grid(True)
        if metrics_tracker.policy_similarity:
            plt.subplot(2, 2, 4)
            plt.plot(metrics_tracker.policy_similarity, color='limegreen')
            plt.xlabel('Update Steps')
            plt.ylabel('Similarity')
            plt.title('Policy Similarity with LLM')
            plt.grid(True)
        plt.tight_layout()
        png_path = os.path.join(self.save_dir, 'training_losses.png')
        pdf_path = os.path.join(self.save_dir, 'training_losses.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_metrics(self, metrics_tracker):
        """绘制综合指标（含风格统一与移动平均）"""
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(metrics_tracker.episode_rewards, color='crimson')
        if len(metrics_tracker.episode_rewards) > 10:
            ma = self._moving_average(metrics_tracker.episode_rewards, min(50, len(metrics_tracker.episode_rewards)//10))
            plt.plot(ma, 'r--', alpha=0.8, label='MA')
            plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.grid(True)
        plt.subplot(2, 2, 2)
        plt.plot(metrics_tracker.episode_delays, color='deepskyblue')
        plt.xlabel('Episode')
        plt.ylabel('Delay')
        plt.title('Task Delay')
        plt.grid(True)
        plt.subplot(2, 2, 3)
        plt.plot(metrics_tracker.episode_energy, color='goldenrod')
        plt.xlabel('Episode')
        plt.ylabel('Energy')
        plt.title('Energy Consumption')
        plt.grid(True)
        plt.subplot(2, 2, 4)
        window_size = 10
        llm_usage_moving_avg = []
        for i in range(len(metrics_tracker.llm_used)):
            if i < window_size:
                llm_usage_moving_avg.append(np.mean(metrics_tracker.llm_used[:i+1]))
            else:
                llm_usage_moving_avg.append(np.mean(metrics_tracker.llm_used[i-window_size+1:i+1]))
        plt.plot(llm_usage_moving_avg, color='limegreen')
        plt.xlabel('Episode')
        plt.ylabel('Usage Ratio')
        plt.title('LLM Usage Ratio (Moving Avg)')
        plt.grid(True)
        plt.tight_layout()
        png_path = os.path.join(self.save_dir, 'metrics.png')
        pdf_path = os.path.join(self.save_dir, 'metrics.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_training_curves(episode_rewards, episode_latencies, episode_energies,
                        episode_completion_rates=None, training_losses=None, save_dir="results"):
    """绘制训练曲线 - A2C-MEC 风格优化版"""
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LLM4RL 训练曲线', fontsize=16, fontweight='bold')
    if episode_rewards:
        axes[0, 0].plot(episode_rewards, color='crimson', linestyle='-', marker='^', markersize=4, label='Reward')
        axes[0, 0].set_title('Episode 奖励', fontsize=12)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('总奖励')
        axes[0, 0].grid(True, alpha=0.3)
        if len(episode_rewards) > 10:
            window_size = min(50, max(10, len(episode_rewards) // 10))
            moving_avg = []
            for i in range(len(episode_rewards)):
                start_idx = max(0, i - window_size + 1)
                moving_avg.append(np.mean(episode_rewards[start_idx:i+1]))
            axes[0, 0].plot(moving_avg, 'r--', linewidth=1, alpha=0.8, label=f'移动平均({window_size})')
            axes[0, 0].legend()
    if episode_latencies:
        axes[0, 1].plot(episode_latencies, color='deepskyblue', linestyle='-', marker='o', markersize=3)
        axes[0, 1].set_title('平均时延', fontsize=12)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('时延 (s)')
        axes[0, 1].grid(True, alpha=0.3)
    if episode_energies:
        axes[1, 0].plot(episode_energies, color='goldenrod', linestyle='-', marker='s', markersize=3)
        axes[1, 0].set_title('平均能耗', fontsize=12)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('能耗 (J)')
        axes[1, 0].grid(True, alpha=0.3)
    if episode_completion_rates:
        axes[1, 1].plot(episode_completion_rates, color='limegreen', linestyle='-', marker='*', markersize=4)
        axes[1, 1].set_title('任务完成率', fontsize=12)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('完成率')
        axes[1, 1].grid(True, alpha=0.3)
    elif training_losses:
        axes[1, 1].plot(training_losses, 'm-', linewidth=2)
        axes[1, 1].set_title('训练损失', fontsize=12)
        axes[1, 1].set_xlabel('更新步数')
        axes[1, 1].set_ylabel('损失')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].set_title('训练指标', fontsize=12)
    plt.tight_layout()
    png_path = os.path.join(save_dir, 'training_curves.png')
    pdf_path = os.path.join(save_dir, 'training_curves.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存至: {png_path}")
    return png_path

def plot_comparison_from_csv(csv_files, metric_column='Latency_s', labels=None, save_dir="results", title=None, xlabel=None, ylabel=None):
    """从CSV绘制多算法对比曲线，仿照A2C-MEC风格"""
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    import pandas as pd
    colors = ['crimson', 'goldenrod', 'limegreen', 'deepskyblue', 'purple']
    markers = ['^', 'o', 's', '*', '+']
    plt.figure()
    for idx, csv_path in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            x = df['Episode'].tolist() if 'Episode' in df.columns else list(range(1, len(df)+1))
            y = df[metric_column].tolist()
            label = labels[idx] if labels and idx < len(labels) else os.path.splitext(os.path.basename(csv_path))[0]
            plt.plot(x, y, label=label, color=colors[idx % len(colors)], linestyle='-', marker=markers[idx % len(markers)], linewidth=2, markersize=6)
        except Exception:
            continue
    if xlabel:
        plt.xlabel(xlabel)
    else:
        plt.xlabel('Episode')
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(metric_column)
    if title:
        plt.title(title)
    plt.legend(ncol=2)
    plt.grid(True)
    png_path = os.path.join(save_dir, f'comparison_{metric_column}.png')
    pdf_path = os.path.join(save_dir, f'comparison_{metric_column}.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    return png_path