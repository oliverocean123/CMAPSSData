import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 深度学习导入
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout, Input, MultiHeadAttention, LayerNormalization, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CompleteRULAnalysis:
    def __init__(self, dataset_name='FD001', window_size=30, max_rul=125):
        self.dataset_name = dataset_name
        self.window_size = window_size
        self.max_rul = max_rul
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.best_params = {}
        
    def load_data(self):
        """加载数据集"""
        cols = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
               [f'sensor_{i}' for i in range(1, 22)]
        
        self.train_df = pd.read_csv(f'train_{self.dataset_name}.txt', sep=r'\s+', header=None, names=cols)
        self.test_df = pd.read_csv(f'test_{self.dataset_name}.txt', sep=r'\s+', header=None, names=cols)
        self.rul_df = pd.read_csv(f'RUL_{self.dataset_name}.txt', sep=r'\s+', header=None, names=['RUL'])
        
        print(f"数据集 {self.dataset_name} 加载完成:")
        print(f"训练集: {self.train_df.shape}, 测试集: {self.test_df.shape}")
        
    def preprocess_data(self):
        """数据预处理"""
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
        print(f"使用特征数量: {len(self.feature_cols)}")
        
        # 标准化
        self.train_df[self.feature_cols] = self.scaler.fit_transform(self.train_df[self.feature_cols])
        self.test_df[self.feature_cols] = self.scaler.transform(self.test_df[self.feature_cols])
        
    def extract_features(self, window):
        """提取统计特征"""
        features = []
        for col in range(window.shape[1]):
            col_data = window[:, col]
            if len(col_data) == 0:
                features.extend([0] * 10)
                continue
                
            features.extend([
                np.mean(col_data),
                np.std(col_data),
                np.max(col_data),
                np.min(col_data),
                col_data[-1],  # 最后值
                col_data[-1] - col_data[0] if len(col_data) > 1 else 0,  # 变化量
                np.median(col_data),
                np.percentile(col_data, 75) - np.percentile(col_data, 25),  # 四分位距
                np.var(col_data),  # 方差
                np.sum(np.abs(np.diff(col_data))) / max(len(col_data) - 1, 1)  # 平均绝对变化
            ])
        return features
    
    def create_windows(self, df, is_train=True):
        """创建滑动窗口特征"""
        X = []
        y = []
        
        for unit in df['unit_number'].unique():
            unit_data = df[df['unit_number'] == unit][self.feature_cols]
            unit_rul = df[df['unit_number'] == unit]['RUL'] if is_train else None
            
            if len(unit_data) < self.window_size:
                # 填充不足的数据
                if len(unit_data) > 0:
                    padding_needed = self.window_size - len(unit_data)
                    last_row = unit_data.iloc[-1:].values
                    padding = np.tile(last_row, (padding_needed, 1))
                    unit_data_array = np.vstack([unit_data.values, padding])
                else:
                    unit_data_array = np.zeros((self.window_size, len(self.feature_cols)))
                    
                if is_train and unit_rul is not None and len(unit_rul) > 0:
                    last_rul = unit_rul.iloc[-1]
                    padding_rul = [last_rul] * padding_needed
                    unit_rul = pd.concat([unit_rul, pd.Series(padding_rul)], ignore_index=True)
            else:
                unit_data_array = unit_data.values
            
            if is_train:
                # 训练集：创建滑动窗口
                for i in range(len(unit_data_array) - self.window_size + 1):
                    window = unit_data_array[i:i+self.window_size]
                    X.append(window)
                    if unit_rul is not None and len(unit_rul) > i+self.window_size-1:
                        y.append(unit_rul.iloc[i+self.window_size-1])
                    else:
                        y.append(0)
            else:
                # 测试集：只取最后一个窗口
                window = unit_data_array[-self.window_size:]
                X.append(window)
        
        return np.array(X), np.array(y) if is_train else None
    
    def feature_engineering(self):
        """特征工程"""
        self.X_train_seq, self.y_train = self.create_windows(self.train_df, True)
        self.X_test_seq, _ = self.create_windows(self.test_df, False)
        self.y_test = self.rul_df['RUL'].clip(upper=self.max_rul).values
        
        # 提取统计特征
        self.X_train_stat = np.array([self.extract_features(w) for w in self.X_train_seq])
        self.X_test_stat = np.array([self.extract_features(w) for w in self.X_test_seq])
        
        print(f"序列特征: 训练集 {self.X_train_seq.shape}, 测试集 {self.X_test_seq.shape}")
        print(f"统计特征: 训练集 {self.X_train_stat.shape}, 测试集 {self.X_test_stat.shape}")
    
    def compute_phm_score(self, y_true, y_pred):
        """计算PHM Score"""
        score = 0
        for true, pred in zip(y_true, y_pred):
            if pred < true:
                score += np.exp(-(true - pred) / 13) - 1
            else:
                score += np.exp((pred - true) / 10) - 1
        return score
    
    # ========== 机器学习模型 ==========
    def train_xgboost(self):
        """XGBoost"""
        print("训练XGBoost...")
        param_dist = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [4, 6, 8],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(xgb_model, param_dist, n_iter=12, cv=3, 
                                  scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42)
        search.fit(self.X_train_stat, self.y_train)
        
        self.models['XGBoost'] = search.best_estimator_
        self.best_params['XGBoost'] = search.best_params_
        
    def train_lightgbm(self):
        """LightGBM"""
        print("训练LightGBM...")
        param_dist = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [4, 6, 8],
            'num_leaves': [31, 63, 127],
            'subsample': [0.8, 0.9]
        }
        
        lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
        search = RandomizedSearchCV(lgb_model, param_dist, n_iter=12, cv=3,
                                  scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42)
        search.fit(self.X_train_stat, self.y_train)
        
        self.models['LightGBM'] = search.best_estimator_
        self.best_params['LightGBM'] = search.best_params_
        
    def train_random_forest(self):
        """随机森林"""
        print("训练随机森林...")
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(rf, param_dist, n_iter=10, cv=3,
                                  scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42)
        search.fit(self.X_train_stat, self.y_train)
        
        self.models['RandomForest'] = search.best_estimator_
        self.best_params['RandomForest'] = search.best_params_
        
    def train_svr(self):
        """支持向量回归"""
        print("训练SVR...")
        param_dist = {
            'C': [1, 10, 100],
            'gamma': ['scale', 0.01, 0.1],
            'epsilon': [0.01, 0.1, 0.2]
        }
        
        svr = SVR(kernel='rbf')
        search = RandomizedSearchCV(svr, param_dist, n_iter=8, cv=3,
                                  scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42)
        search.fit(self.X_train_stat, self.y_train)
        
        self.models['SVR'] = search.best_estimator_
        self.best_params['SVR'] = search.best_params_
    
    # ========== 深度学习模型 ==========
    def build_lstm_model(self, input_shape):
        """构建LSTM模型"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def build_gru_model(self, input_shape):
        """构建GRU模型"""
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def build_bidirectional_lstm_model(self, input_shape):
        """构建双向LSTM模型"""
        model = Sequential([
            Bidirectional(LSTM(32, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(16, return_sequences=False)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def build_cnn_lstm_model(self, input_shape):
        """构建CNN-LSTM模型"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def build_transformer_model(self, input_shape):
        """构建Transformer模型"""
        inputs = Input(shape=input_shape)
        
        # Multi-head attention
        attention = MultiHeadAttention(key_dim=32, num_heads=4, dropout=0.1)
        x = attention(inputs, inputs)
        x = Dropout(0.1)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs
        
        # Feed forward network
        x = Dense(64, activation="relu")(res)
        x = Dropout(0.1)(x)
        x = Dense(inputs.shape[-1])(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = x + res
        
        # Output layers
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def build_temporal_fusion_transformer(self, input_shape):
        """构建简化版Temporal Fusion Transformer"""
        inputs = Input(shape=input_shape)
        
        # Variable selection network (简化版)
        x = Dense(input_shape[-1], activation='sigmoid')(inputs)
        x = x * inputs  # 特征选择
        
        # LSTM encoder
        lstm_out = LSTM(64, return_sequences=True)(x)
        lstm_out = Dropout(0.2)(lstm_out)
        
        # Multi-head attention
        attention = MultiHeadAttention(key_dim=32, num_heads=4, dropout=0.1)
        attn_out = attention(lstm_out, lstm_out)
        attn_out = LayerNormalization()(attn_out)
        
        # Gate mechanism
        gate = Dense(64, activation='sigmoid')(attn_out)
        gated_out = gate * attn_out
        
        # Output
        x = Flatten()(gated_out)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train_deep_models(self):
        """训练所有深度学习模型"""
        print("开始训练深度学习模型...")
        input_shape = (self.window_size, len(self.feature_cols))
        
        # 准备回调函数
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor='val_loss')
        ]
        
        # LSTM
        print("训练LSTM...")
        lstm_model = self.build_lstm_model(input_shape)
        lstm_model.fit(self.X_train_seq, self.y_train, epochs=50, batch_size=32, 
                      validation_split=0.2, callbacks=callbacks, verbose=0)
        self.models['LSTM'] = lstm_model
        
        # GRU
        print("训练GRU...")
        gru_model = self.build_gru_model(input_shape)
        gru_model.fit(self.X_train_seq, self.y_train, epochs=50, batch_size=32,
                     validation_split=0.2, callbacks=callbacks, verbose=0)
        self.models['GRU'] = gru_model
        
        # Bidirectional LSTM
        print("训练双向LSTM...")
        bi_lstm_model = self.build_bidirectional_lstm_model(input_shape)
        bi_lstm_model.fit(self.X_train_seq, self.y_train, epochs=50, batch_size=32,
                         validation_split=0.2, callbacks=callbacks, verbose=0)
        self.models['BiLSTM'] = bi_lstm_model
        
        # CNN-LSTM
        print("训练CNN-LSTM...")
        cnn_lstm_model = self.build_cnn_lstm_model(input_shape)
        cnn_lstm_model.fit(self.X_train_seq, self.y_train, epochs=50, batch_size=32,
                          validation_split=0.2, callbacks=callbacks, verbose=0)
        self.models['CNN-LSTM'] = cnn_lstm_model
        
        # Transformer
        print("训练Transformer...")
        transformer_model = self.build_transformer_model(input_shape)
        transformer_model.fit(self.X_train_seq, self.y_train, epochs=50, batch_size=32,
                             validation_split=0.2, callbacks=callbacks, verbose=0)
        self.models['Transformer'] = transformer_model
        
        # Temporal Fusion Transformer (简化版)
        print("训练TFT...")
        tft_model = self.build_temporal_fusion_transformer(input_shape)
        tft_model.fit(self.X_train_seq, self.y_train, epochs=50, batch_size=32,
                     validation_split=0.2, callbacks=callbacks, verbose=0)
        self.models['TFT'] = tft_model
    
    def train_all_models(self):
        """训练所有模型"""
        print("\n开始训练所有模型...")
        
        # 机器学习模型
        self.train_xgboost()
        self.train_lightgbm()
        self.train_random_forest()
        self.train_svr()
        
        # 深度学习模型
        self.train_deep_models()
        
        print(f"\n训练完成！共训练了 {len(self.models)} 个模型")
    
    def evaluate_models(self):
        """评估所有模型"""
        print("\n开始评估模型...")
        print(f"{'模型':<15} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'PHM Score':<12} {'准确率':<8}")
        print("-" * 75)
        
        for name, model in self.models.items():
            # 选择合适的特征
            if name in ['XGBoost', 'LightGBM', 'RandomForest', 'SVR']:
                X_test = self.X_test_stat
            else:
                X_test = self.X_test_seq
            
            # 预测
            if name in ['XGBoost', 'LightGBM', 'RandomForest', 'SVR']:
                y_pred = model.predict(X_test)
            else:
                y_pred = model.predict(X_test, verbose=0)
            if len(y_pred.shape) > 1:
                y_pred = y_pred.flatten()
            
            # 计算指标
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            phm_score = self.compute_phm_score(self.y_test, y_pred)
            
            # 计算准确率（容忍度为15%）
            tolerance = 0.15
            accuracy = np.mean(np.abs(self.y_test - y_pred) / np.maximum(self.y_test, 1) <= tolerance) * 100
            
            self.results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'PHM_Score': phm_score,
                'Accuracy': accuracy
            }
            
            print(f"{name:<15} {mae:<8.2f} {rmse:<8.2f} {r2:<8.3f} {phm_score:<12.2f} {accuracy:<8.1f}%")
    
    def plot_results(self):
        """可视化结果"""
        if not self.results:
            print("没有结果可以绘制")
            return
            
        # 创建结果DataFrame
        df = pd.DataFrame(self.results).T
        
        # 绘制指标对比
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        metrics = ['MAE', 'RMSE', 'R2', 'PHM_Score', 'Accuracy']
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple']
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                values = df[metric].sort_values(ascending=(metric != 'R2' and metric != 'Accuracy'))
                ax = axes[i]
                bars = ax.bar(range(len(values)), values, color=colors[i % len(colors)])
                ax.set_title(f'{metric} 对比', fontsize=14, fontweight='bold')
                ax.set_xticks(range(len(values)))
                ax.set_xticklabels(values.index, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                for j, (bar, val) in enumerate(zip(bars, values)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 隐藏多余的子图
        if len(metrics) < len(axes):
            axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.dataset_name}_complete_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
    
    def get_top_models(self, top_n=6):
        """获取表现最好的前N个模型"""
        if not self.results:
            return None
            
        df = pd.DataFrame(self.results).T
        
        # 综合评分
        phm_norm = 1 - (df['PHM_Score'] - df['PHM_Score'].min()) / (df['PHM_Score'].max() - df['PHM_Score'].min() + 1e-8)
        r2_norm = (df['R2'] - df['R2'].min()) / (df['R2'].max() - df['R2'].min() + 1e-8)
        acc_norm = (df['Accuracy'] - df['Accuracy'].min()) / (df['Accuracy'].max() - df['Accuracy'].min() + 1e-8)
        
        df['综合评分'] = 0.4 * phm_norm + 0.3 * r2_norm + 0.3 * acc_norm
        
        top_models = df.nlargest(top_n, '综合评分')
        
        print(f"\n前{top_n}个最优模型（综合评分）:")
        print(f"{'排名':<4} {'模型':<15} {'综合评分':<10} {'PHM Score':<12} {'R²':<8} {'准确率':<8}")
        print("-" * 75)
        
        for i, (model, row) in enumerate(top_models.iterrows(), 1):
            print(f"{i:<4} {model:<15} {row['综合评分']:<10.3f} {row['PHM_Score']:<12.2f} {row['R2']:<8.3f} {row['Accuracy']:<8.1f}%")
        
        return top_models
    
    def save_results(self):
        """保存结果"""
        if not self.results:
            return
            
        # 保存详细结果
        df = pd.DataFrame(self.results).T
        df.to_csv(f'{self.dataset_name}_complete_results.csv')
        
        # 保存最优参数
        if self.best_params:
            import json
            with open(f'{self.dataset_name}_complete_best_params.json', 'w', encoding='utf-8') as f:
                json.dump(self.best_params, f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到 {self.dataset_name}_complete_results.csv")
    
    def run_analysis(self):
        """运行完整分析"""
        print(f"开始完整分析数据集: {self.dataset_name}")
        
        self.load_data()
        self.preprocess_data()
        self.feature_engineering()
        self.train_all_models()
        self.evaluate_models()
        results_df = self.plot_results()
        top_models = self.get_top_models(6)
        self.save_results()
        
        return results_df, top_models

def run_multiple_datasets(datasets=['FD001', 'FD002', 'FD003', 'FD004']):
    """运行多个数据集的完整分析"""
    all_results = {}
    all_top_models = {}
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"处理数据集: {dataset}")
        print(f"{'='*80}")
        
        try:
            analyzer = CompleteRULAnalysis(dataset_name=dataset)
            results_df, top_models = analyzer.run_analysis()
            all_results[dataset] = results_df
            all_top_models[dataset] = top_models
        except Exception as e:
            print(f"处理数据集 {dataset} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 汇总结果
    if all_results:
        print(f"\n{'='*100}")
        print("所有数据集完整结果汇总")
        print(f"{'='*100}")
        
        # 计算平均指标
        all_dfs = list(all_results.values())
        avg_results = pd.concat(all_dfs).groupby(level=0).mean()
        
        print("\n各模型平均表现:")
        print(f"{'模型':<15} {'平均MAE':<10} {'平均RMSE':<10} {'平均R²':<10} {'平均PHM':<12} {'平均准确率':<10}")
        print("-" * 85)
        
        for model in avg_results.index:
            row = avg_results.loc[model]
            print(f"{model:<15} {row['MAE']:<10.2f} {row['RMSE']:<10.2f} {row['R2']:<10.3f} {row['PHM_Score']:<12.2f} {row['Accuracy']:<10.1f}%")
        
        # 最终排名
        phm_ranking = avg_results['PHM_Score'].sort_values()
        
        print(f"\n最终模型排名 (基于平均PHM Score):")
        for i, (model, score) in enumerate(phm_ranking.items(), 1):
            r2_score = avg_results.loc[model, 'R2']
            accuracy = avg_results.loc[model, 'Accuracy']
            print(f"{i:2d}. {model:<15} - PHM: {score:8.2f}, R²: {r2_score:6.3f}, 准确率: {accuracy:5.1f}%")
        
        # 保存汇总结果
        avg_results.to_csv('complete_analysis_average_results.csv')
        print(f"\n完整分析汇总结果已保存到 complete_analysis_average_results.csv")
        
        return all_results, avg_results, all_top_models
    
    return None, None, None

if __name__ == "__main__":
    print("开始完整RUL预测分析（包含所有深度学习模型）...")
    
    # 先测试单个数据集
    analyzer = CompleteRULAnalysis(dataset_name='FD001')
    results_df, top_models = analyzer.run_analysis()
    
    print("\n如需运行所有数据集，请取消下面的注释:")
    print("# all_results, avg_results, all_top_models = run_multiple_datasets()")