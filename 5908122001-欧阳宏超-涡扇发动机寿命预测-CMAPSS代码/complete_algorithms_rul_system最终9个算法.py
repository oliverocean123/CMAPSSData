"""
完整的七算法RUL预测系统
包含：SVR, GRU, BiLSTM, Transformer, Full TFT, XGBoost, LightGBM
特点：
- 完整的数据处理和特征工程
- 参数优化和配置管理
- 详细的可视化和分析
- 清晰的算法逻辑和模块化设计
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
import time
import json
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

# PyTorch imports for deep learning models
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
    print(f"✅ PyTorch {torch.__version__} 可用")
    if torch.cuda.is_available():
        print(f"🚀 CUDA可用: {torch.cuda.get_device_name(0)}")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("❌ PyTorch未安装，将跳过深度学习模型")

# 设置随机种子
np.random.seed(42)
if PYTORCH_AVAILABLE:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CompleteSevenAlgorithmsRULSystem:
    """完整的七算法RUL预测系统"""
    
    def __init__(self, dataset='FD001', window_size=30, max_rul=125, device='cuda'):
        self.dataset = dataset
        self.window_size = window_size
        self.max_rul = max_rul
        self.device = torch.device(device if torch.cuda.is_available() and PYTORCH_AVAILABLE else 'cpu')
        
        # 数据存储
        self.train_df = None
        self.test_df = None
        self.rul_df = None
        self.feature_cols = None
        
        # 不同类型的数据
        self.X_train_stat = None  # 统计特征 - 用于传统ML
        self.X_test_stat = None
        self.X_train_seq = None   # 序列数据 - 用于深度学习
        self.X_test_seq = None
        self.y_train = None
        self.y_test = None
        
        # 标准化器
        self.scaler_stat = StandardScaler()
        self.scaler_seq = StandardScaler()
        
        # 结果存储
        self.models = {}
        self.results = {}
        self.best_params = {}
        self.training_history = {}
        
        # 创建输出目录
        self.output_dir = Path(f'results_{dataset}')
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🎯 完整七算法RUL预测系统初始化")
        print(f"📊 数据集: {dataset}")
        print(f"🤖 算法: SVR, GRU, BiLSTM, Transformer, Full TFT, XGBoost, LightGBM")
        print(f"💻 设备: {self.device}")
        print(f"📁 输出目录: {self.output_dir}")
    
    # ========== 数据处理模块 ==========
    def load_data(self):
        """加载数据"""
        print(f"📊 加载 {self.dataset} 数据集...")
        
        cols = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
               [f'sensor_{i}' for i in range(1, 22)]
        
        self.train_df = pd.read_csv(f'train_{self.dataset}.txt', sep=r'\s+', header=None, names=cols)
        self.test_df = pd.read_csv(f'test_{self.dataset}.txt', sep=r'\s+', header=None, names=cols)
        self.rul_df = pd.read_csv(f'RUL_{self.dataset}.txt', sep=r'\s+', header=None, names=['RUL'])
        
        print(f"✅ 数据加载完成")
        print(f"   训练集: {self.train_df.shape}")
        print(f"   测试集: {self.test_df.shape}")
        print(f"   RUL标签: {self.rul_df.shape}")
    
    def preprocess_data(self):
        """数据预处理"""
        print(f"🔄 数据预处理...")
        
        # 计算RUL
        def calculate_rul(group):
            group = group.copy()
            group['RUL'] = group['time_cycles'].max() - group['time_cycles']
            group['RUL'] = group['RUL'].clip(upper=self.max_rul)
            return group
        
        self.train_df = self.train_df.groupby('unit_number').apply(calculate_rul).reset_index(drop=True)
        
        # 选择有用的传感器
        sensor_cols = [col for col in self.train_df.columns if 'sensor' in col]
        std_values = self.train_df[sensor_cols].std()
        useful_sensors = std_values[std_values > 0].index.tolist()
        
        self.feature_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + useful_sensors
        
        print(f"✅ 数据预处理完成")
        print(f"   特征数: {len(self.feature_cols)}")
        print(f"   有用传感器: {len(useful_sensors)}")
    
    def extract_comprehensive_features(self, window):
        """提取comprehensive版本的完整统计特征 - 12个特征/列"""
        features = []
        for col in range(window.shape[1]):
            col_data = window[:, col]
            if len(col_data) == 0:
                features.extend([0] * 12)
                continue
                
            features.extend([
                np.mean(col_data),                    # 均值
                np.std(col_data),                     # 标准差
                np.max(col_data),                     # 最大值
                np.min(col_data),                     # 最小值
                np.ptp(col_data),                     # 极差
                np.median(col_data),                  # 中位数
                np.percentile(col_data, 25),          # 25分位数
                np.percentile(col_data, 75),          # 75分位数
                np.sum(np.diff(col_data) > 0) / max(len(col_data) - 1, 1) if len(col_data) > 1 else 0,  # 上升趋势比例
                np.var(col_data),                     # 方差
                np.sum(np.abs(np.diff(col_data))) / max(len(col_data) - 1, 1) if len(col_data) > 1 else 0,  # 平均绝对变化
                col_data[-1] - col_data[0] if len(col_data) > 1 else 0,  # 总变化量
            ])
        return features
    
    def create_statistical_data(self, df, is_train=True):
        """创建统计特征数据 - 用于传统机器学习算法"""
        X = []
        y = []
        
        for unit in df['unit_number'].unique():
            unit_data = df[df['unit_number'] == unit][self.feature_cols].values
            unit_rul = df[df['unit_number'] == unit]['RUL'].values if is_train else None
            
            # 处理数据长度不足的情况
            if len(unit_data) < self.window_size:
                if len(unit_data) > 0:
                    padding_needed = self.window_size - len(unit_data)
                    last_row = unit_data[-1:]
                    padding = np.tile(last_row, (padding_needed, 1))
                    unit_data = np.vstack([unit_data, padding])
                    
                    if is_train and unit_rul is not None:
                        last_rul = unit_rul[-1]
                        unit_rul = np.concatenate([unit_rul, [last_rul] * padding_needed])
                else:
                    unit_data = np.zeros((self.window_size, len(self.feature_cols)))
                    if is_train:
                        unit_rul = np.zeros(self.window_size)
            
            if is_train:
                # 训练集：创建滑动窗口
                for i in range(len(unit_data) - self.window_size + 1):
                    window = unit_data[i:i+self.window_size]
                    X.append(self.extract_comprehensive_features(window))
                    if unit_rul is not None and len(unit_rul) > i+self.window_size-1:
                        y.append(unit_rul[i+self.window_size-1])
            else:
                # 测试集：只取最后一个窗口
                window = unit_data[-self.window_size:]
                X.append(self.extract_comprehensive_features(window))
        
        return np.array(X), np.array(y) if is_train else None
    
    def create_sequence_data(self, df, is_train=True):
        """创建真正的序列数据 - 用于深度学习算法"""
        X = []
        y = []
        
        for unit in df['unit_number'].unique():
            unit_data = df[df['unit_number'] == unit][self.feature_cols].values
            unit_rul = df[df['unit_number'] == unit]['RUL'].values if is_train else None
            
            # 处理数据长度不足的情况
            if len(unit_data) < self.window_size:
                if len(unit_data) > 0:
                    padding_needed = self.window_size - len(unit_data)
                    last_row = unit_data[-1:]
                    padding = np.tile(last_row, (padding_needed, 1))
                    unit_data = np.vstack([unit_data, padding])
                    
                    if is_train and unit_rul is not None:
                        last_rul = unit_rul[-1]
                        unit_rul = np.concatenate([unit_rul, [last_rul] * padding_needed])
                else:
                    unit_data = np.zeros((self.window_size, len(self.feature_cols)))
                    if is_train:
                        unit_rul = np.zeros(self.window_size)
            
            if is_train:
                # 训练集：创建滑动窗口，保持序列结构
                for i in range(len(unit_data) - self.window_size + 1):
                    window = unit_data[i:i+self.window_size]  # 保持 (window_size, n_features) 形状
                    X.append(window)
                    if unit_rul is not None and len(unit_rul) > i+self.window_size-1:
                        y.append(unit_rul[i+self.window_size-1])
            else:
                # 测试集：只取最后一个窗口
                window = unit_data[-self.window_size:]  # (window_size, n_features)
                X.append(window)
        
        return np.array(X), np.array(y) if is_train else None
    
    def feature_engineering(self):
        """特征工程 - 为不同算法准备正确的数据格式"""
        print(f"🔧 特征工程...")
        
        # 1. 创建统计特征数据 (用于SVR, XGBoost, LightGBM)
        print(f"📊 为传统ML算法创建统计特征数据...")
        self.X_train_stat, self.y_train = self.create_statistical_data(self.train_df, is_train=True)
        self.X_test_stat, _ = self.create_statistical_data(self.test_df, is_train=False)
        self.y_test = self.rul_df['RUL'].clip(upper=self.max_rul).values
        
        # 2. 创建序列数据 (用于GRU, BiLSTM, Transformer, TFT)
        print(f"📊 为深度学习算法创建序列数据...")
        X_train_seq_raw, y_train_seq = self.create_sequence_data(self.train_df, is_train=True)
        X_test_seq_raw, _ = self.create_sequence_data(self.test_df, is_train=False)
        
        # 3. 标准化统计特征
        self.X_train_stat = self.scaler_stat.fit_transform(self.X_train_stat)
        self.X_test_stat = self.scaler_stat.transform(self.X_test_stat)
        
        # 4. 标准化序列数据
        n_samples, n_timesteps, n_features = X_train_seq_raw.shape
        X_train_flat = X_train_seq_raw.reshape(-1, n_features)
        X_train_scaled = self.scaler_seq.fit_transform(X_train_flat)
        self.X_train_seq = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
        
        n_samples_test = X_test_seq_raw.shape[0]
        X_test_flat = X_test_seq_raw.reshape(-1, n_features)
        X_test_scaled = self.scaler_seq.transform(X_test_flat)
        self.X_test_seq = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)
        
        print(f"✅ 特征工程完成:")
        print(f"   统计特征 (传统ML): 训练{self.X_train_stat.shape}, 测试{self.X_test_stat.shape}")
        print(f"   序列数据 (深度学习): 训练{self.X_train_seq.shape}, 测试{self.X_test_seq.shape}")
        print(f"   目标变量: 训练{self.y_train.shape}, 测试{self.y_test.shape}")
    
    # ========== 评分计算模块 ==========
    def calculate_nasa_score(self, y_true, y_pred):
        """计算NASA评分"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        a1 = 13  # 早预测惩罚参数
        a2 = 10  # 晚预测惩罚参数
        
        d_i = np.abs(y_true - y_pred)
        early_mask = y_pred < y_true
        late_mask = y_pred >= y_true
        
        scores = np.zeros_like(d_i, dtype=float)
        
        if np.any(early_mask):
            scores[early_mask] = np.exp(-d_i[early_mask] / a1) - 1
        
        if np.any(late_mask):
            scores[late_mask] = np.exp(d_i[late_mask] / a2) - 1
        
        return np.sum(scores)
    
    def calculate_phm_score(self, y_true, y_pred):
        """计算PHM评分"""
        nasa_score = self.calculate_nasa_score(y_true, y_pred)
        return nasa_score / len(y_true)
    
    # ========== 传统机器学习算法模块 ==========
    def train_svr(self):
        """训练SVR算法"""
        print(f"🤖 训练SVR算法...")
        start_time = time.time()
        
        # 参数网格
        param_grid = {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1, 'scale', 'auto'],
            'epsilon': [0.01, 0.1, 0.2, 0.5, 1.0]
        }
        
        print(f"   🔍 执行网格搜索...")
        svr = SVR()
        grid_search = GridSearchCV(
            svr, param_grid, cv=3, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train_stat, self.y_train)
        
        # 获取最佳模型
        best_svr = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # 预测
        y_pred = best_svr.predict(self.X_test_stat)
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        nasa_score = self.calculate_nasa_score(self.y_test, y_pred)
        phm_score = self.calculate_phm_score(self.y_test, y_pred)
        
        training_time = time.time() - start_time
        
        # 保存结果
        self.models['SVR'] = best_svr
        self.best_params['SVR'] = best_params
        self.results['SVR'] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'nasa_score': nasa_score,
            'phm_score': phm_score,
            'training_time': training_time,
            'predictions': y_pred,
            'best_params': best_params
        }
        
        print(f"✅ SVR训练完成!")
        print(f"   🏆 最佳参数: {best_params}")
        print(f"   📊 RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.4f}")
        print(f"   🎯 NASA Score: {nasa_score:.3f}, PHM Score: {phm_score:.4f}")
        print(f"   ⏱️ 训练时间: {training_time:.1f}秒")
    
    def train_xgboost(self):
        """训练XGBoost算法"""
        print(f"🤖 训练XGBoost算法...")
        start_time = time.time()
        
        # 参数网格
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        print(f"   🔍 执行随机搜索...")
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        random_search = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=20, cv=3,
            scoring='neg_mean_squared_error', 
            n_jobs=-1, random_state=42
        )
        
        random_search.fit(self.X_train_stat, self.y_train)
        
        # 获取最佳模型
        best_xgb = random_search.best_estimator_
        best_params = random_search.best_params_
        
        # 预测
        y_pred = best_xgb.predict(self.X_test_stat)
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        nasa_score = self.calculate_nasa_score(self.y_test, y_pred)
        phm_score = self.calculate_phm_score(self.y_test, y_pred)
        
        training_time = time.time() - start_time
        
        # 保存结果
        self.models['XGBoost'] = best_xgb
        self.best_params['XGBoost'] = best_params
        self.results['XGBoost'] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'nasa_score': nasa_score,
            'phm_score': phm_score,
            'training_time': training_time,
            'predictions': y_pred,
            'best_params': best_params
        }
        
        print(f"✅ XGBoost训练完成!")
        print(f"   🏆 最佳参数: {best_params}")
        print(f"   📊 RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.4f}")
        print(f"   🎯 NASA Score: {nasa_score:.3f}, PHM Score: {phm_score:.4f}")
        print(f"   ⏱️ 训练时间: {training_time:.1f}秒")
    
    def train_lightgbm(self):
        """训练LightGBM算法"""
        print(f"🤖 训练LightGBM算法...")
        start_time = time.time()
        
        # 参数网格
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'num_leaves': [31, 50, 100],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        print(f"   🔍 执行随机搜索...")
        lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
        random_search = RandomizedSearchCV(
            lgb_model, param_grid, n_iter=20, cv=3,
            scoring='neg_mean_squared_error', 
            n_jobs=-1, random_state=42
        )
        
        random_search.fit(self.X_train_stat, self.y_train)
        
        # 获取最佳模型
        best_lgb = random_search.best_estimator_
        best_params = random_search.best_params_
        
        # 预测
        y_pred = best_lgb.predict(self.X_test_stat)
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        nasa_score = self.calculate_nasa_score(self.y_test, y_pred)
        phm_score = self.calculate_phm_score(self.y_test, y_pred)
        
        training_time = time.time() - start_time
        
        # 保存结果
        self.models['LightGBM'] = best_lgb
        self.best_params['LightGBM'] = best_params
        self.results['LightGBM'] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'nasa_score': nasa_score,
            'phm_score': phm_score,
            'training_time': training_time,
            'predictions': y_pred,
            'best_params': best_params
        }
        
        print(f"✅ LightGBM训练完成!")
        print(f"   🏆 最佳参数: {best_params}")
        print(f"   📊 RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.4f}")
        print(f"   🎯 NASA Score: {nasa_score:.3f}, PHM Score: {phm_score:.4f}")
        print(f"   ⏱️ 训练时间: {training_time:.1f}秒")
    
    # ========== 深度学习算法模块 ==========
    
    # GRU模型定义
    class OptimizedGRUModel(nn.Module):
        """优化的GRU模型"""
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3, 
                     use_batch_norm=True, use_residual=False):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.use_batch_norm = use_batch_norm
            self.use_residual = use_residual
            
            self.gru = nn.GRU(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                             bidirectional=False)
            
            self.dropout = nn.Dropout(dropout)
            
            if use_batch_norm:
                self.batch_norm = nn.BatchNorm1d(hidden_size)
            
            # 全连接层
            self.fc_layers = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 4, 1)
            )
            
            # 残差连接
            if use_residual:
                self.residual_fc = nn.Linear(input_size, 1)
            
        def forward(self, x):
            gru_out, _ = self.gru(x)
            last_output = gru_out[:, -1, :]
            
            if self.use_batch_norm and last_output.size(0) > 1:
                last_output = self.batch_norm(last_output)
            
            output = self.fc_layers(last_output)
            
            # 残差连接
            if self.use_residual:
                residual = self.residual_fc(x.mean(dim=1))
                output = output + residual
            
            return output
    
    # BiLSTM模型定义
    class OptimizedBiLSTMModel(nn.Module):
        """优化的双向LSTM模型"""
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3,
                     use_batch_norm=True, use_attention=False):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.use_batch_norm = use_batch_norm
            self.use_attention = use_attention
            
            self.bilstm = nn.LSTM(input_size, hidden_size, num_layers,
                                 batch_first=True, dropout=dropout if num_layers > 1 else 0,
                                 bidirectional=True)
            
            self.dropout = nn.Dropout(dropout)
            
            # 双向LSTM输出维度是hidden_size * 2
            lstm_output_size = hidden_size * 2
            
            if use_batch_norm:
                self.batch_norm = nn.BatchNorm1d(lstm_output_size)
            
            # 注意力机制
            if use_attention:
                self.attention = nn.MultiheadAttention(lstm_output_size, num_heads=8, dropout=dropout)
                self.layer_norm = nn.LayerNorm(lstm_output_size)
            
            # 全连接层
            self.fc_layers = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_output_size // 4, 1)
            )
            
        def forward(self, x):
            lstm_out, _ = self.bilstm(x)
            
            if self.use_attention:
                # 使用注意力机制
                lstm_out_transposed = lstm_out.transpose(0, 1)  # (seq_len, batch, features)
                attn_out, _ = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
                attn_out = attn_out.transpose(0, 1)  # (batch, seq_len, features)
                attn_out = self.layer_norm(attn_out + lstm_out)
                last_output = attn_out[:, -1, :]
            else:
                last_output = lstm_out[:, -1, :]
            
            if self.use_batch_norm and last_output.size(0) > 1:
                last_output = self.batch_norm(last_output)
            
            output = self.fc_layers(last_output)
            return output
    
    # Transformer模型定义
    class OptimizedTransformerModel(nn.Module):
        """优化的Transformer模型"""
        def __init__(self, input_size, d_model=128, nhead=8, num_layers=4, dropout=0.3,
                     use_positional_encoding=True, use_layer_norm=True):
            super().__init__()
            self.input_size = input_size
            self.d_model = d_model
            self.use_positional_encoding = use_positional_encoding
            self.use_layer_norm = use_layer_norm
            
            # 输入投影
            self.input_projection = nn.Linear(input_size, d_model)
            
            # 位置编码
            if use_positional_encoding:
                self.pos_encoding = nn.Parameter(torch.randn(1000, d_model) * 0.1)
            
            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
            # 输出层
            self.dropout = nn.Dropout(dropout)
            
            if use_layer_norm:
                self.layer_norm = nn.LayerNorm(d_model)
            
            self.fc_layers = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, 1)
            )
            
        def forward(self, x):
            batch_size, seq_len, _ = x.shape
            
            # 输入投影
            x = self.input_projection(x)
            
            # 添加位置编码
            if self.use_positional_encoding:
                x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
            
            # Transformer编码
            x = self.transformer(x)
            
            # 全局平均池化 + 最后时间步
            global_avg = x.mean(dim=1)
            last_step = x[:, -1, :]
            
            # 组合特征
            combined = global_avg + last_step
            
            if self.use_layer_norm:
                combined = self.layer_norm(combined)
            
            combined = self.dropout(combined)
            
            # 输出
            output = self.fc_layers(combined)
            return output
    
    # Full TFT模型定义
    class FullTemporalFusionTransformer(nn.Module):
        """完整版Temporal Fusion Transformer - 增强版实现"""
        def __init__(self, input_size, hidden_size=128, num_heads=8, num_layers=3, dropout=0.3):
            super().__init__()
            
            # 导入增强版TFT
            try:
                from enhanced_tft_model import EnhancedTemporalFusionTransformer
                
                # 使用增强版TFT作为核心
                self.enhanced_tft = EnhancedTemporalFusionTransformer(
                    seq_input_size=input_size,
                    static_input_size=None,  # 在这个版本中不使用静态特征
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dropout=dropout,
                    use_static_features=False
                )
                self.use_enhanced = True
                print("✅ 使用增强版TFT实现")
                
            except ImportError:
                print("⚠️ 增强版TFT不可用，使用基础实现")
                self.use_enhanced = False
                self._build_basic_tft(input_size, hidden_size, num_heads, num_layers, dropout)
        
        def _build_basic_tft(self, input_size, hidden_size, num_heads, num_layers, dropout):
            """构建基础TFT实现（回退方案）"""
            self.input_size = input_size
            self.hidden_size = hidden_size
            
            # Variable Selection Network
            self.variable_selection = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size),
                nn.Sigmoid()
            )
            
            # LSTM Encoder-Decoder
            self.lstm_encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.lstm_decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            
            # Multi-Head Attention
            self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
            
            # Gated Residual Network
            self.grn1 = self._build_grn(hidden_size)
            self.grn2 = self._build_grn(hidden_size)
            
            # Static Enrichment
            self.static_enrichment = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # Temporal Self-Attention
            self.temporal_attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
            
            # Position-wise Feed Forward
            self.feed_forward = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            
            # Output layers
            self.layer_norm = nn.LayerNorm(hidden_size)
            self.output_projection = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            )
            
        def _build_grn(self, input_size):
            """构建Gated Residual Network"""
            return nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ELU(),
                nn.Linear(input_size, input_size),
                nn.Dropout(0.1),
                nn.Linear(input_size, input_size * 2)  # 用于门控机制
            )
        
        def _apply_grn(self, x, grn):
            """应用Gated Residual Network"""
            grn_output = grn(x)
            gate, transform = torch.chunk(grn_output, 2, dim=-1)
            gate = torch.sigmoid(gate)
            return gate * transform + (1 - gate) * x
        
        def forward(self, x):
            if self.use_enhanced:
                # 使用增强版TFT
                outputs = self.enhanced_tft(x, None)
                return outputs['prediction']
            else:
                # 使用基础实现
                return self._forward_basic(x)
        
        def _forward_basic(self, x):
            """基础TFT前向传播"""
            batch_size, seq_len, _ = x.shape
            
            # Variable Selection
            variable_weights = self.variable_selection(x)
            x_selected = x * variable_weights
            
            # LSTM Encoding
            lstm_out, (h_n, c_n) = self.lstm_encoder(x_selected)
            
            # Static Enrichment
            enriched = self.static_enrichment(lstm_out)
            
            # Temporal Self-Attention
            # 转换为 (seq_len, batch_size, hidden_size) for attention
            attn_input = enriched.transpose(0, 1)
            attn_output, _ = self.temporal_attention(attn_input, attn_input, attn_input)
            attn_output = attn_output.transpose(0, 1)  # 转回 (batch_size, seq_len, hidden_size)
            
            # Apply GRN
            grn_output = self._apply_grn(attn_output, self.grn1)
            
            # Position-wise Feed Forward
            ff_output = self.feed_forward(grn_output)
            
            # Apply second GRN
            grn_output2 = self._apply_grn(ff_output, self.grn2)
            
            # Layer normalization
            normalized = self.layer_norm(grn_output2)
            
            # Global average pooling
            pooled = torch.mean(normalized, dim=1)
            
            # Final prediction
            output = self.output_projection(pooled)
            
            return output
            static_context = self.static_enrichment(h_n[-1])  # 使用最后一个隐藏状态
            
            # 将静态上下文广播到所有时间步
            static_context_expanded = static_context.unsqueeze(1).expand(-1, seq_len, -1)
            enriched_lstm = lstm_out + static_context_expanded
            
            # Gated Residual Network 1
            grn1_out = self._apply_grn(enriched_lstm, self.grn1)
            
            # Temporal Self-Attention
            grn1_transposed = grn1_out.transpose(0, 1)  # (seq_len, batch, hidden)
            attn_out, _ = self.temporal_attention(grn1_transposed, grn1_transposed, grn1_transposed)
            attn_out = attn_out.transpose(0, 1)  # (batch, seq_len, hidden)
            
            # Residual connection
            attn_out = self.layer_norm(attn_out + grn1_out)
            
            # Gated Residual Network 2
            grn2_out = self._apply_grn(attn_out, self.grn2)
            
            # Position-wise Feed Forward
            ff_out = self.feed_forward(grn2_out)
            
            # Final residual connection
            final_out = self.layer_norm(ff_out + grn2_out)
            
            # Global average pooling + last timestep
            global_avg = final_out.mean(dim=1)
            last_timestep = final_out[:, -1, :]
            combined = global_avg + last_timestep
            
            # Output projection
            output = self.output_projection(combined)
            return output
    
    def train_pytorch_model(self, model_class, model_name, **model_kwargs):
        """训练PyTorch深度学习模型的通用函数"""
        if not PYTORCH_AVAILABLE:
            print(f"❌ PyTorch不可用，跳过{model_name}模型")
            return
        
        print(f"🧠 训练{model_name}模型...")
        start_time = time.time()
        
        # 使用正确的序列数据
        X_train = torch.FloatTensor(self.X_train_seq).to(self.device)
        y_train = torch.FloatTensor(self.y_train).to(self.device)
        X_test = torch.FloatTensor(self.X_test_seq).to(self.device)
        
        # 验证集分割
        val_size = int(0.2 * len(X_train))
        indices = torch.randperm(len(X_train))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train_split = X_train[train_indices]
        y_train_split = y_train[train_indices]
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        
        # 创建数据加载器
        batch_size = model_kwargs.get('batch_size', 32)
        train_dataset = TensorDataset(X_train_split, y_train_split)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # 创建模型
        input_size = self.X_train_seq.shape[2]
        model_params = {k: v for k, v in model_kwargs.items() if k not in ['lr', 'batch_size', 'epochs', 'weight_decay']}
        model = model_class(input_size=input_size, **model_params).to(self.device)
        
        # 优化器和调度器
        lr = model_kwargs.get('lr', 0.001)
        weight_decay = model_kwargs.get('weight_decay', 0.01)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        criterion = nn.MSELoss()
        
        # 早停参数
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        # 训练历史
        train_losses = []
        val_losses = []
        
        print(f"   📊 训练集: {len(train_indices)}, 验证集: {len(val_indices)}")
        print(f"   🏗️ 模型参数: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练循环
        epochs = model_kwargs.get('epochs', 100)
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            train_batches = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), self.output_dir / f'best_{model_name.lower()}_model.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            if epoch % 20 == 0 or patience_counter == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch:3d}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={current_lr:.6f}")
            
            # 早停
            if patience_counter >= patience:
                print(f"   ⏹️ 早停触发 (Epoch {epoch})")
                break
        
        # 加载最佳模型
        model.load_state_dict(torch.load(self.output_dir / f'best_{model_name.lower()}_model.pth'))
        
        # 最终预测
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).squeeze().cpu().numpy()
        
        # 计算指标
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        nasa_score = self.calculate_nasa_score(self.y_test, y_pred)
        phm_score = self.calculate_phm_score(self.y_test, y_pred)
        
        training_time = time.time() - start_time
        
        # 保存结果
        self.models[model_name] = model
        self.best_params[model_name] = model_kwargs
        self.results[model_name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'nasa_score': nasa_score,
            'phm_score': phm_score,
            'training_time': training_time,
            'predictions': y_pred,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_params': model_kwargs
        }
        
        # 保存训练历史
        self.training_history[model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        print(f"✅ {model_name}训练完成!")
        print(f"   📊 RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.4f}")
        print(f"   🎯 NASA Score: {nasa_score:.3f}, PHM Score: {phm_score:.4f}")
        print(f"   ⏱️ 训练时间: {training_time:.1f}秒")
        
        return model, y_pred
    
    def train_gru(self):
        """训练GRU模型"""
        self.train_pytorch_model(
            self.OptimizedGRUModel, 'GRU',
            hidden_size=64, num_layers=2, dropout=0.3,
            use_batch_norm=True, use_residual=False,
            lr=0.001, batch_size=32, epochs=100, weight_decay=0.01
        )
    
    def train_bilstm(self):
        """训练BiLSTM模型"""
        self.train_pytorch_model(
            self.OptimizedBiLSTMModel, 'BiLSTM',
            hidden_size=64, num_layers=2, dropout=0.3,
            use_batch_norm=True, use_attention=True,
            lr=0.001, batch_size=32, epochs=100, weight_decay=0.01
        )
    
    def train_transformer(self):
        """训练Transformer模型"""
        self.train_pytorch_model(
            self.OptimizedTransformerModel, 'Transformer',
            d_model=128, nhead=8, num_layers=4, dropout=0.3,
            use_positional_encoding=True, use_layer_norm=True,
            lr=0.001, batch_size=32, epochs=100, weight_decay=0.01
        )
    
    def train_full_tft(self):
        """训练Full TFT模型"""
        self.train_pytorch_model(
            self.FullTemporalFusionTransformer, 'Full_TFT',
            hidden_size=128, num_heads=8, num_layers=3, dropout=0.3,
            lr=0.0005, batch_size=32, epochs=120, weight_decay=0.01
        )
    
    # ========== 训练控制模块 ==========
    def train_all_models(self):
        """训练所有七个算法"""
        print(f"🚀 开始训练所有七个算法...")
        print("="*80)
        
        # 1. 传统机器学习算法
        print(f"\n📊 训练传统机器学习算法...")
        self.train_svr()
        self.train_xgboost()
        self.train_lightgbm()
        
        # 2. 深度学习算法
        if PYTORCH_AVAILABLE:
            print(f"\n🧠 训练深度学习算法...")
            self.train_gru()
            self.train_bilstm()
            self.train_transformer()
            self.train_full_tft()
        else:
            print("❌ PyTorch不可用，跳过深度学习模型")
        
        print(f"\n🎉 所有算法训练完成！共训练了 {len(self.models)} 个模型")
    
    # ========== 可视化模块 ==========
    def plot_individual_predictions(self):
        """为每个算法绘制单独的预测RUL和真实RUL对比图"""
        print(f"📊 生成各算法单独预测对比图...")
        
        if not self.results:
            print("❌ 没有结果可以绘制")
            return
        
        # 为每个算法创建单独的图
        for model_name in self.results.keys():
            y_pred = self.results[model_name]['predictions']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{model_name} 算法预测结果分析 - {self.dataset}', fontsize=16, fontweight='bold')
            
            # 1. 预测vs真实值散点图
            ax1 = axes[0, 0]
            ax1.scatter(self.y_test, y_pred, alpha=0.6, s=50, color='blue')
            ax1.plot([0, self.max_rul], [0, self.max_rul], 'r--', lw=2, label='Perfect Prediction')
            ax1.set_xlabel('真实RUL', fontsize=12)
            ax1.set_ylabel('预测RUL', fontsize=12)
            ax1.set_title('预测值 vs 真实值', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 添加性能指标文本
            rmse = self.results[model_name]['rmse']
            r2 = self.results[model_name]['r2']
            phm = self.results[model_name]['phm_score']
            ax1.text(0.05, 0.95, f'RMSE: {rmse:.3f}\nR²: {r2:.4f}\nPHM: {phm:.3f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 2. 时间序列预测对比图
            ax2 = axes[0, 1]
            sample_indices = range(len(self.y_test))
            ax2.plot(sample_indices, self.y_test, 'b-', label='True RUL', linewidth=2, alpha=0.8)
            ax2.plot(sample_indices, y_pred, 'r-', label='Predicted RUL', linewidth=2, alpha=0.8)
            ax2.set_xlabel('Sample Index', fontsize=12)
            ax2.set_ylabel('RUL', fontsize=12)
            ax2.set_title('True vs Predicted RUL Time Series', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. 残差分布图
            ax3 = axes[1, 0]
            residuals = y_pred - self.y_test
            ax3.scatter(self.y_test, residuals, alpha=0.6, s=50, color='green')
            ax3.axhline(y=0, color='r', linestyle='--', lw=2)
            ax3.set_xlabel('真实RUL', fontsize=12)
            ax3.set_ylabel('残差 (预测值 - 真实值)', fontsize=12)
            ax3.set_title('残差分布图', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # 4. 误差分布直方图
            ax4 = axes[1, 1]
            ax4.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax4.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
            ax4.set_xlabel('残差', fontsize=12)
            ax4.set_ylabel('频次', fontsize=12)
            ax4.set_title('残差分布直方图', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{model_name}_individual_prediction_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        print(f"✅ 各算法单独预测对比图已保存到 {self.output_dir}")
    
    def plot_training_curves(self):
        """绘制各深度学习算法的训练曲线"""
        print(f"📊 生成训练曲线图...")
        
        if not self.training_history:
            print("❌ 没有训练历史数据")
            return
        
        # 深度学习模型
        dl_models = [name for name in self.training_history.keys() if name in ['GRU', 'BiLSTM', 'Transformer', 'Full_TFT']]
        
        if not dl_models:
            print("❌ 没有深度学习模型的训练历史")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'深度学习算法训练曲线 - {self.dataset}', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, model_name in enumerate(dl_models[:4]):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            history = self.training_history[model_name]
            train_losses = history['train_losses']
            val_losses = history['val_losses']
            epochs = range(1, len(train_losses) + 1)
            
            ax.plot(epochs, train_losses, label='Training Loss', color=colors[i], linewidth=2, alpha=0.8)
            ax.plot(epochs, val_losses, label='Validation Loss', color=colors[i], linewidth=2, linestyle='--', alpha=0.8)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'{model_name} 训练曲线', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 标记最佳epoch
            best_epoch = np.argmin(val_losses) + 1
            best_val_loss = min(val_losses)
            ax.scatter(best_epoch, best_val_loss, color='red', s=100, marker='*', zorder=5)
            ax.text(best_epoch, best_val_loss, f'Best: {best_val_loss:.4f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'training_curves_{self.dataset}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 训练曲线图已保存到 {self.output_dir}")
    
    def plot_metrics_comparison(self):
        """绘制各算法各指标的对比图"""
        print(f"📊 生成指标对比图...")
        
        if not self.results:
            print("❌ 没有结果可以绘制")
            return
        
        # 创建结果DataFrame
        df_results = pd.DataFrame(self.results).T
        
        # 设置图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'七算法性能指标对比 - {self.dataset}', fontsize=16, fontweight='bold')
        
        metrics = ['rmse', 'mae', 'r2', 'nasa_score', 'phm_score', 'training_time']
        metric_names = ['RMSE', 'MAE', 'R²', 'NASA Score', 'PHM Score', 'Training Time (s)']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            values = df_results[metric].sort_values(ascending=(metric not in ['r2']))
            bars = ax.bar(range(len(values)), values, color=colors[i], alpha=0.8)
            
            # 添加数值标签
            for j, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values.index, rotation=45, ha='right')
            ax.set_ylabel(name, fontsize=12)
            ax.set_title(f'{name} 对比', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # 高亮最佳值
            if metric == 'r2':
                best_idx = len(values) - 1  # R²越大越好
            else:
                best_idx = 0  # 其他指标越小越好（除了training_time是信息性的）
            
            if metric != 'training_time':  # 训练时间不需要高亮最佳
                bars[best_idx].set_color('red')
                bars[best_idx].set_alpha(1.0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'metrics_comparison_{self.dataset}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 指标对比图已保存到 {self.output_dir}")
    
    def plot_comprehensive_analysis(self):
        """绘制综合分析图"""
        print(f"📊 生成综合分析图...")
        
        if not self.results:
            print("❌ 没有结果可以绘制")
            return
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'七算法RUL预测系统综合分析 - {self.dataset}', fontsize=18, fontweight='bold')
        
        # 1. 算法性能雷达图
        ax1 = plt.subplot(3, 3, 1, projection='polar')
        
        # 准备雷达图数据
        algorithms = list(self.results.keys())
        metrics = ['rmse', 'mae', 'phm_score']  # 选择关键指标
        
        # 标准化指标（越小越好的指标需要反转）
        df_results = pd.DataFrame(self.results).T
        normalized_data = {}
        
        for metric in metrics:
            values = df_results[metric].values
            # 标准化到0-1，越小越好的指标反转
            normalized = 1 - (values - values.min()) / (values.max() - values.min() + 1e-8)
            normalized_data[metric] = normalized
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        
        for i, alg in enumerate(algorithms):
            values = [normalized_data[metric][i] for metric in metrics]
            values += values[:1]  # 闭合
            
            ax1.plot(angles, values, 'o-', linewidth=2, label=alg, color=colors[i])
            ax1.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(['RMSE', 'MAE', 'PHM Score'])
        ax1.set_ylim(0, 1)
        ax1.set_title('算法性能雷达图', fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 2. 最佳算法预测散点图
        ax2 = plt.subplot(3, 3, 2)
        best_alg = min(self.results.keys(), key=lambda x: self.results[x]['phm_score'])
        best_pred = self.results[best_alg]['predictions']
        
        ax2.scatter(self.y_test, best_pred, alpha=0.6, s=50, color='blue')
        ax2.plot([0, self.max_rul], [0, self.max_rul], 'r--', lw=2)
        ax2.set_xlabel('真实RUL')
        ax2.set_ylabel('预测RUL')
        ax2.set_title(f'最佳算法预测效果 ({best_alg})')
        ax2.grid(True, alpha=0.3)
        
        # 3. 算法排名条形图
        ax3 = plt.subplot(3, 3, 3)
        phm_scores = df_results['phm_score'].sort_values()
        bars = ax3.barh(range(len(phm_scores)), phm_scores.values, color='lightcoral')
        ax3.set_yticks(range(len(phm_scores)))
        ax3.set_yticklabels(phm_scores.index)
        ax3.set_xlabel('PHM Score')
        ax3.set_title('算法PHM Score排名')
        ax3.grid(axis='x', alpha=0.3)
        
        # 标记最佳
        bars[0].set_color('red')
        
        # 4-6. 各指标详细对比
        metrics_detail = ['rmse', 'r2', 'training_time']
        metric_names_detail = ['RMSE', 'R² Score', 'Training Time (s)']
        
        for i, (metric, name) in enumerate(zip(metrics_detail, metric_names_detail)):
            ax = plt.subplot(3, 3, 4 + i)
            values = df_results[metric].sort_values(ascending=(metric != 'r2'))
            bars = ax.bar(range(len(values)), values, alpha=0.8)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values.index, rotation=45, ha='right')
            ax.set_ylabel(name)
            ax.set_title(f'{name} 对比')
            ax.grid(axis='y', alpha=0.3)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 7. 预测误差分布
        ax7 = plt.subplot(3, 3, 7)
        for i, alg in enumerate(algorithms):
            pred = self.results[alg]['predictions']
            errors = np.abs(pred - self.y_test)
            ax7.hist(errors, bins=20, alpha=0.5, label=alg, density=True)
        
        ax7.set_xlabel('绝对误差')
        ax7.set_ylabel('密度')
        ax7.set_title('预测误差分布')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. 算法复杂度vs性能
        ax8 = plt.subplot(3, 3, 8)
        training_times = [self.results[alg]['training_time'] for alg in algorithms]
        phm_scores_list = [self.results[alg]['phm_score'] for alg in algorithms]
        
        scatter = ax8.scatter(training_times, phm_scores_list, s=100, alpha=0.7, c=range(len(algorithms)), cmap='viridis')
        
        for i, alg in enumerate(algorithms):
            ax8.annotate(alg, (training_times[i], phm_scores_list[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax8.set_xlabel('训练时间 (秒)')
        ax8.set_ylabel('PHM Score')
        ax8.set_title('算法复杂度 vs 性能')
        ax8.grid(True, alpha=0.3)
        
        # 9. 数据统计信息
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        stats_text = f"""
数据集统计信息:
• 数据集: {self.dataset}
• 训练样本: {len(self.y_train)}
• 测试样本: {len(self.y_test)}
• 特征维度: {len(self.feature_cols)}
• 窗口大小: {self.window_size}
• 最大RUL: {self.max_rul}

最佳算法: {best_alg}
• RMSE: {self.results[best_alg]['rmse']:.3f}
• R²: {self.results[best_alg]['r2']:.4f}
• PHM Score: {self.results[best_alg]['phm_score']:.3f}
        """
        
        ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'comprehensive_analysis_{self.dataset}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 综合分析图已保存到 {self.output_dir}")
    
    # ========== 数据保存模块 ==========
    def save_results_and_data(self):
        """保存结果和数据"""
        print(f"💾 保存结果和数据...")
        
        # 1. 保存评估结果
        if self.results:
            df_results = pd.DataFrame(self.results).T
            df_results.to_csv(self.output_dir / f'results_{self.dataset}.csv')
            print(f"   ✅ 评估结果已保存到 results_{self.dataset}.csv")
        
        # 2. 保存最佳参数
        if self.best_params:
            with open(self.output_dir / f'best_params_{self.dataset}.json', 'w', encoding='utf-8') as f:
                json.dump(self.best_params, f, indent=2, ensure_ascii=False, default=str)
            print(f"   ✅ 最佳参数已保存到 best_params_{self.dataset}.json")
        
        # 3. 保存训练历史
        if self.training_history:
            with open(self.output_dir / f'training_history_{self.dataset}.json', 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False, default=str)
            print(f"   ✅ 训练历史已保存到 training_history_{self.dataset}.json")
        
        # 4. 保存预测结果
        predictions_data = {}
        predictions_data['y_test'] = self.y_test.tolist()
        for model_name in self.results.keys():
            predictions_data[f'{model_name}_predictions'] = self.results[model_name]['predictions'].tolist()
        
        with open(self.output_dir / f'predictions_{self.dataset}.json', 'w', encoding='utf-8') as f:
            json.dump(predictions_data, f, indent=2, ensure_ascii=False)
        print(f"   ✅ 预测结果已保存到 predictions_{self.dataset}.json")
        
        # 5. 保存模型（传统ML模型）
        for model_name, model in self.models.items():
            if model_name in ['SVR', 'XGBoost', 'LightGBM']:
                with open(self.output_dir / f'{model_name}_model_{self.dataset}.pkl', 'wb') as f:
                    pickle.dump(model, f)
                print(f"   ✅ {model_name}模型已保存到 {model_name}_model_{self.dataset}.pkl")
        
        print(f"💾 所有结果和数据已保存到 {self.output_dir}")
    
    def generate_summary_report(self):
        """生成总结报告"""
        if not self.results:
            print("❌ 没有结果可以生成报告")
            return
        
        print(f"\n" + "="*100)
        print(f"🎯 完整七算法RUL预测系统 - 性能报告")
        print(f"="*100)
        print(f"📊 数据集: {self.dataset}")
        print(f"🔧 窗口大小: {self.window_size}")
        print(f"🎯 最大RUL: {self.max_rul}")
        print(f"💻 设备: {self.device}")
        print(f"📁 输出目录: {self.output_dir}")
        
        # 按PHM Score排序
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['phm_score'])
        
        print(f"\n🏆 算法性能排行榜 (按PHM Score排序):")
        print("-"*100)
        print(f"{'排名':<4} {'算法':<12} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'NASA':<10} {'PHM':<8} {'时间':<8}")
        print("-"*100)
        
        for i, (name, result) in enumerate(sorted_results, 1):
            print(f"{i:<4} {name:<12} {result['rmse']:<8.3f} {result['mae']:<8.3f} "
                  f"{result['r2']:<8.4f} {result['nasa_score']:<10.2f} {result['phm_score']:<8.3f} {result['training_time']:<8.1f}s")
        
        # 最佳算法详细信息
        best_name, best_result = sorted_results[0]
        print(f"\n🥇 最佳算法: {best_name}")
        print(f"   📊 RMSE: {best_result['rmse']:.3f}")
        print(f"   📊 MAE: {best_result['mae']:.3f}")
        print(f"   📊 R²: {best_result['r2']:.4f}")
        print(f"   🎯 NASA Score: {best_result['nasa_score']:.3f}")
        print(f"   🏆 PHM Score: {best_result['phm_score']:.3f}")
        print(f"   ⏱️ 训练时间: {best_result['training_time']:.1f}秒")
        
        # 算法类型分析
        ml_algorithms = ['SVR', 'XGBoost', 'LightGBM']
        dl_algorithms = ['GRU', 'BiLSTM', 'Transformer', 'Full_TFT']
        
        ml_results = {name: result for name, result in self.results.items() if name in ml_algorithms}
        dl_results = {name: result for name, result in self.results.items() if name in dl_algorithms}
        
        if ml_results:
            best_ml = min(ml_results.items(), key=lambda x: x[1]['phm_score'])
            print(f"\n🤖 最佳传统ML算法: {best_ml[0]} (PHM: {best_ml[1]['phm_score']:.3f})")
        
        if dl_results:
            best_dl = min(dl_results.items(), key=lambda x: x[1]['phm_score'])
            print(f"🧠 最佳深度学习算法: {best_dl[0]} (PHM: {best_dl[1]['phm_score']:.3f})")
        
        # 性能统计
        all_phm_scores = [result['phm_score'] for result in self.results.values()]
        all_rmse_scores = [result['rmse'] for result in self.results.values()]
        
        print(f"\n📈 性能统计:")
        print(f"   PHM Score - 最佳: {min(all_phm_scores):.3f}, 最差: {max(all_phm_scores):.3f}, 平均: {np.mean(all_phm_scores):.3f}")
        print(f"   RMSE - 最佳: {min(all_rmse_scores):.3f}, 最差: {max(all_rmse_scores):.3f}, 平均: {np.mean(all_rmse_scores):.3f}")
        
        print(f"\n💾 所有结果已保存到: {self.output_dir}")
        print(f"="*100)
    
    # ========== 主控制模块 ==========
    def run_complete_analysis(self):
        """运行完整的七算法分析"""
        print(f"🚀 开始完整七算法RUL预测系统分析")
        print(f"="*80)
        
        try:
            # 1. 数据处理
            print(f"\n📊 第一阶段：数据处理")
            self.load_data()
            self.preprocess_data()
            self.feature_engineering()
            
            # 2. 模型训练
            print(f"\n🤖 第二阶段：模型训练")
            self.train_all_models()
            
            # 3. 结果可视化
            print(f"\n📊 第三阶段：结果可视化")
            self.plot_individual_predictions()
            self.plot_training_curves()
            self.plot_metrics_comparison()
            self.plot_comprehensive_analysis()
            
            # 4. 数据保存
            print(f"\n💾 第四阶段：数据保存")
            self.save_results_and_data()
            
            # 5. 生成报告
            print(f"\n📋 第五阶段：生成报告")
            self.generate_summary_report()
            
            print(f"\n🎉 完整七算法RUL预测系统分析完成!")
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("🎯 完整七算法RUL预测系统")
    print("🤖 算法: SVR, GRU, BiLSTM, Transformer, Full TFT, XGBoost, LightGBM")
    print("🔧 特色: 完整数据处理, 参数优化, 详细可视化, 模块化设计")
    print("="*80)
    
    # 可以选择运行单个数据集或多个数据集
    datasets = ['FD001']  # 可以扩展为 ['FD001', 'FD002', 'FD003', 'FD004']
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"处理数据集: {dataset}")
        print(f"{'='*80}")
        
        try:
            # 创建系统实例
            system = CompleteSevenAlgorithmsRULSystem(
                dataset=dataset, 
                window_size=30, 
                max_rul=125, 
                device='cuda'
            )
            
            # 运行完整分析
            system.run_complete_analysis()
            
        except Exception as e:
            print(f"❌ 处理数据集 {dataset} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🎉 所有数据集处理完成!")

if __name__ == "__main__":
    main()