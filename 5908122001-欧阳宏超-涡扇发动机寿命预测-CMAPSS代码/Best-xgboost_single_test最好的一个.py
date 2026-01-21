"""
基于enhanced_hybrid_rul_system框架的XGBoost单独测试
使用训练得到的最佳参数进行性能评估
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
from pathlib import Path
import pickle
import json

# 导入必要的库
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class XGBoostSingleTester:
    """XGBoost单独测试器"""
    
    def __init__(self, dataset_name='FD001'):
        self.dataset_name = dataset_name
        self.window_size = 30
        self.max_rul = 125
        self.scaler = StandardScaler()
        
        # 存储结果
        self.model = None
        self.results = {}
        self.training_time = 0
        self.prediction_time = 0
        
        # 创建输出目录
        self.output_dir = Path(f'xgboost_test_results_{dataset_name}')
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🎯 XGBoost单独测试器初始化")
        print(f"📊 数据集: {dataset_name}")
        print(f"📁 输出目录: {self.output_dir}")
    
    def load_and_preprocess_data(self):
        """加载和预处理数据 - 复用enhanced_hybrid_rul_system的逻辑"""
        print(f"📊 加载数据集 {self.dataset_name}...")
        
        # 定义列名
        cols = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
               [f'sensor_{i}' for i in range(1, 22)]
        
        # 加载数据
        self.train_df = pd.read_csv(f'train_{self.dataset_name}.txt', sep=r'\s+', header=None, names=cols)
        self.test_df = pd.read_csv(f'test_{self.dataset_name}.txt', sep=r'\s+', header=None, names=cols)
        self.rul_df = pd.read_csv(f'RUL_{self.dataset_name}.txt', sep=r'\s+', header=None, names=['RUL'])
        
        print(f"✅ 数据加载完成:")
        print(f"   训练集: {self.train_df.shape}")
        print(f"   测试集: {self.test_df.shape}")
        print(f"   RUL标签: {self.rul_df.shape}")
        
        # 计算训练集RUL
        def calculate_rul(group):
            group = group.copy()
            group['RUL'] = group['time_cycles'].max() - group['time_cycles']
            group['RUL'] = group['RUL'].clip(upper=self.max_rul)
            return group
        
        self.train_df = self.train_df.groupby('unit_number').apply(calculate_rul).reset_index(drop=True)
        
        # 特征选择 - 选择有变化的传感器
        sensor_cols = [col for col in self.train_df.columns if 'sensor' in col]
        std_values = self.train_df[sensor_cols].std()
        useful_sensors = std_values[std_values > 0.01].index.tolist()
        
        self.feature_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + useful_sensors
        
        print(f"🔧 特征选择完成，使用 {len(self.feature_cols)} 个特征")
        print(f"   有用传感器: {len(useful_sensors)} 个")
        
        # 数据标准化
        self.train_df[self.feature_cols] = self.scaler.fit_transform(self.train_df[self.feature_cols])
        self.test_df[self.feature_cols] = self.scaler.transform(self.test_df[self.feature_cols])
        
        print(f"✅ 数据预处理完成")
    
    def create_statistical_features(self):
        """创建统计特征 - 复用enhanced_hybrid_rul_system的逻辑"""
        print(f"🔧 创建统计特征...")
        
        # 1. 创建序列特征（用于提取统计特征）
        X_train_seq, self.y_train = self._create_sequences(self.train_df, True)
        X_test_seq, _ = self._create_sequences(self.test_df, False)
        self.y_test = self.rul_df['RUL'].clip(upper=self.max_rul).values
        
        # 2. 从序列特征提取统计特征
        self.X_train_stat = np.array([self._extract_statistical_features(w) for w in X_train_seq])
        self.X_test_stat = np.array([self._extract_statistical_features(w) for w in X_test_seq])
        
        print(f"✅ 统计特征创建完成:")
        print(f"   训练特征: {self.X_train_stat.shape}")
        print(f"   测试特征: {self.X_test_stat.shape}")
        print(f"   训练标签: {self.y_train.shape}")
        print(f"   测试标签: {self.y_test.shape}")
        print(f"   特征维度: {self.X_train_stat.shape[1]}")
    
    def _create_sequences(self, df, is_train=True):
        """创建序列数据"""
        X, y = [], []
        
        for unit in df['unit_number'].unique():
            unit_data = df[df['unit_number'] == unit][self.feature_cols].values
            unit_rul = df[df['unit_number'] == unit]['RUL'].values if is_train else None
            
            # 处理短序列
            if len(unit_data) < self.window_size:
                if len(unit_data) > 0:
                    padding_needed = self.window_size - len(unit_data)
                    padding = np.tile(unit_data[-1:], (padding_needed, 1))
                    unit_data = np.vstack([unit_data, padding])
                    if is_train:
                        unit_rul = np.concatenate([unit_rul, [unit_rul[-1]] * padding_needed])
                else:
                    continue
            
            if is_train:
                for i in range(len(unit_data) - self.window_size + 1):
                    X.append(unit_data[i:i+self.window_size])
                    y.append(unit_rul[i+self.window_size-1])
            else:
                X.append(unit_data[-self.window_size:])
        
        return np.array(X), np.array(y) if is_train else None
    
    def _extract_statistical_features(self, window):
        """提取统计特征"""
        features = []
        
        for col_idx in range(window.shape[1]):
            col_data = window[:, col_idx]
            
            # 基础统计特征
            features.extend([
                np.mean(col_data),                    # 均值
                np.std(col_data),                     # 标准差
                np.max(col_data),                     # 最大值
                np.min(col_data),                     # 最小值
                np.median(col_data),                  # 中位数
                col_data[-1],                         # 最后值
                col_data[0],                          # 第一值
                col_data[-1] - col_data[0],          # 总变化量
                np.var(col_data),                     # 方差
                np.percentile(col_data, 75) - np.percentile(col_data, 25),  # 四分位距
            ])
            
            # 趋势特征
            if len(col_data) > 1:
                x = np.arange(len(col_data))
                slope = np.polyfit(x, col_data, 1)[0]
                features.append(slope)
                features.append(np.mean(np.abs(np.diff(col_data))))
            else:
                features.extend([0, 0])
        
        return features
    
    def calculate_nasa_score(self, y_true, y_pred):
        """计算NASA评分"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        a1 = 13  # 早预测惩罚参数
        a2 = 10  # 晚预测惩罚参数
        
        scores = []
        for true_val, pred_val in zip(y_true, y_pred):
            d = pred_val - true_val
            if d < 0:  # 早预测
                score = np.exp(-d / a1) - 1
            else:  # 晚预测
                score = np.exp(d / a2) - 1
            scores.append(score)
        
        return np.sum(scores)
    
    def calculate_phm_score(self, y_true, y_pred):
        """计算PHM评分"""
        nasa_score = self.calculate_nasa_score(y_true, y_pred)
        return nasa_score / len(y_true)
    
    def train_xgboost_with_best_params(self):
        """使用最佳参数训练XGBoost"""
        print(f"\n🤖 使用最佳参数训练XGBoost...")
        
        # 最佳参数配置（基于您提供的参数）
        best_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 500,
            'learning_rate': 0.02,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.95,
            'reg_alpha': 0.1,
            'reg_lambda': 4,
            'random_state': 42,
            'n_jobs': -1
        }
        
        print(f"   🏆 使用参数:")
        for key, value in best_params.items():
            print(f"      {key}: {value}")
        
        # 创建模型
        self.model = xgb.XGBRegressor(**best_params)
        
        # 训练计时
        print(f"   🚀 开始训练...")
        start_time = time.time()
        
        self.model.fit(self.X_train_stat, self.y_train)
        
        self.training_time = time.time() - start_time
        
        print(f"✅ XGBoost训练完成!")
        print(f"   ⏱️ 训练时间: {self.training_time:.3f}秒")
        print(f"   🌳 树的数量: {self.model.n_estimators}")
        print(f"   📊 输入特征数: {self.X_train_stat.shape[1]}")
        print(f"   🎯 最大深度: {self.model.max_depth}")
    
    def evaluate_model(self):
        """评估模型性能"""
        print(f"\n📊 评估模型性能...")
        
        # 预测计时
        start_time = time.time()
        y_pred = self.model.predict(self.X_test_stat)
        self.prediction_time = time.time() - start_time
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        nasa_score = self.calculate_nasa_score(self.y_test, y_pred)
        phm_score = self.calculate_phm_score(self.y_test, y_pred)
        
        # 存储结果
        self.results = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'nasa_score': nasa_score,
            'phm_score': phm_score,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'predictions': y_pred,
            'n_features': self.X_train_stat.shape[1],
            'n_train_samples': len(self.y_train),
            'n_test_samples': len(self.y_test)
        }
        
        print(f"✅ 模型评估完成!")
        print(f"   📊 RMSE: {rmse:.3f}")
        print(f"   📊 MAE: {mae:.3f}")
        print(f"   📊 R²: {r2:.4f}")
        print(f"   🎯 NASA Score: {nasa_score:.3f}")
        print(f"   🎯 PHM Score: {phm_score:.4f}")
        print(f"   ⏱️ 预测时间: {self.prediction_time:.3f}秒")
        print(f"   ⚡ 预测速度: {len(self.y_test)/self.prediction_time:.1f} 样本/秒")
    
    def plot_results(self):
        """绘制结果图表"""
        print(f"📊 生成结果图表...")
        
        y_pred = self.results['predictions']
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'XGBoost单独测试结果 - {self.dataset_name}', fontsize=16, fontweight='bold')
        
        # 1. 预测vs真实值散点图
        ax1 = axes[0, 0]
        ax1.scatter(self.y_test, y_pred, alpha=0.6, s=50, color='blue', edgecolors='navy', linewidth=0.5)
        ax1.plot([0, self.max_rul], [0, self.max_rul], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('真实RUL', fontsize=12, fontweight='bold')
        ax1.set_ylabel('预测RUL', fontsize=12, fontweight='bold')
        ax1.set_title('预测值 vs 真实值', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加性能指标文本
        metrics_text = f'RMSE: {self.results["rmse"]:.3f}\nMAE: {self.results["mae"]:.3f}\n' + \
                      f'R²: {self.results["r2"]:.4f}\nNASA: {self.results["nasa_score"]:.2f}\n' + \
                      f'PHM: {self.results["phm_score"]:.4f}'
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
                verticalalignment='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 2. 时间序列预测对比图
        ax2 = axes[0, 1]
        sample_indices = range(len(self.y_test))
        ax2.plot(sample_indices, self.y_test, 'b-', label='True RUL', linewidth=2, alpha=0.8)
        ax2.plot(sample_indices, y_pred, 'r-', label='Predicted RUL', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('RUL', fontsize=12, fontweight='bold')
        ax2.set_title('RUL预测时间序列对比', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 预测误差分布
        ax3 = axes[1, 0]
        errors = y_pred - self.y_test
        ax3.hist(errors, bins=30, alpha=0.7, color='green', edgecolor='darkgreen')
        ax3.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean Error: {np.mean(errors):.3f}')
        ax3.set_xlabel('预测误差 (预测值 - 真实值)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('频次', fontsize=12, fontweight='bold')
        ax3.set_title('预测误差分布', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 性能指标雷达图
        ax4 = axes[1, 1]
        
        # 性能指标（标准化到0-1）
        metrics_names = ['R²', 'RMSE\n(反向)', 'MAE\n(反向)', 'PHM\n(反向)']
        
        # 标准化指标值（越大越好）
        r2_norm = max(0, self.results['r2'])  # R²本身就是0-1
        rmse_norm = max(0, 1 - self.results['rmse'] / 50)  # 假设RMSE=50为最差
        mae_norm = max(0, 1 - self.results['mae'] / 40)   # 假设MAE=40为最差
        phm_norm = max(0, 1 - self.results['phm_score'] / 10)  # 假设PHM=10为最差
        
        values = [r2_norm, rmse_norm, mae_norm, phm_norm]
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        ax4.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.8)
        ax4.fill(angles, values, alpha=0.25, color='blue')
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics_names)
        ax4.set_ylim(0, 1)
        ax4.set_title('性能指标雷达图', fontsize=14, fontweight='bold')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = self.output_dir / f'xgboost_test_results_{self.dataset_name}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ 结果图表已保存: {plot_file}")
    
    def save_results(self):
        """保存结果"""
        print(f"💾 保存测试结果...")
        
        # 保存模型
        model_file = self.output_dir / f'xgboost_best_model_{self.dataset_name}.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # 保存结果JSON
        results_to_save = self.results.copy()
        results_to_save['predictions'] = results_to_save['predictions'].tolist()
        
        results_file = self.output_dir / f'xgboost_test_results_{self.dataset_name}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        # 保存详细报告
        report_file = self.output_dir / f'xgboost_test_report_{self.dataset_name}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"XGBoost单独测试报告 - {self.dataset_name}\n")
            f.write("="*50 + "\n\n")
            f.write(f"数据集信息:\n")
            f.write(f"  训练样本数: {self.results['n_train_samples']}\n")
            f.write(f"  测试样本数: {self.results['n_test_samples']}\n")
            f.write(f"  特征维度: {self.results['n_features']}\n\n")
            f.write(f"模型参数:\n")
            f.write(f"  树的数量: {self.model.n_estimators}\n")
            f.write(f"  最大深度: {self.model.max_depth}\n")
            f.write(f"  学习率: {self.model.learning_rate}\n")
            f.write(f"  子采样率: {self.model.subsample}\n")
            f.write(f"  特征采样率: {self.model.colsample_bytree}\n\n")
            f.write(f"性能指标:\n")
            f.write(f"  RMSE: {self.results['rmse']:.4f}\n")
            f.write(f"  MAE: {self.results['mae']:.4f}\n")
            f.write(f"  R²: {self.results['r2']:.4f}\n")
            f.write(f"  NASA Score: {self.results['nasa_score']:.4f}\n")
            f.write(f"  PHM Score: {self.results['phm_score']:.4f}\n\n")
            f.write(f"时间性能:\n")
            f.write(f"  训练时间: {self.results['training_time']:.3f}秒\n")
            f.write(f"  预测时间: {self.results['prediction_time']:.3f}秒\n")
            f.write(f"  预测速度: {len(self.y_test)/self.results['prediction_time']:.1f} 样本/秒\n")
        
        print(f"   ✅ 模型已保存: {model_file}")
        print(f"   ✅ 结果已保存: {results_file}")
        print(f"   ✅ 报告已保存: {report_file}")
    
    def run_complete_test(self):
        """运行完整测试"""
        print(f"🚀 开始XGBoost完整测试")
        print("="*60)
        
        # 1. 数据处理
        self.load_and_preprocess_data()
        self.create_statistical_features()
        
        # 2. 训练模型
        self.train_xgboost_with_best_params()
        
        # 3. 评估性能
        self.evaluate_model()
        
        # 4. 生成图表
        self.plot_results()
        
        # 5. 保存结果
        self.save_results()
        
        print(f"\n🎉 XGBoost测试完成！")
        print(f"📁 所有结果已保存到: {self.output_dir}")
        print(f"\n📊 最终性能摘要:")
        print(f"   R² Score: {self.results['r2']:.4f}")
        print(f"   RMSE: {self.results['rmse']:.3f}")
        print(f"   训练时间: {self.results['training_time']:.3f}秒")
        print(f"   预测速度: {len(self.y_test)/self.results['prediction_time']:.1f} 样本/秒")

def main():
    """主函数"""
    print("🎯 XGBoost单独性能测试")
    print("="*40)
    
    # 创建测试器
    tester = XGBoostSingleTester(dataset_name='FD001')
    
    # 运行完整测试
    tester.run_complete_test()

if __name__ == "__main__":
    main()