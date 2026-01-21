import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, GroupKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告: LightGBM 未安装，将跳过 LightGBM 模型")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("警告: CatBoost 未安装，将跳过 CatBoost 模型")
import warnings
warnings.filterwarnings('ignore')

# 深度学习相关导入
try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout, Input, MultiHeadAttention, LayerNormalization, BatchNormalization, Bidirectional
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("警告: TensorFlow/Keras 未安装，将跳过深度学习模型")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveRULPrediction:
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
        # 定义列名
        cols = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
               [f'sensor_{i}' for i in range(1, 22)]
        
        # 读取数据
        self.train_df = pd.read_csv(f'train_{self.dataset_name}.txt', sep=r'\s+', header=None, names=cols)
        self.test_df = pd.read_csv(f'test_{self.dataset_name}.txt', sep=r'\s+', header=None, names=cols)
        self.rul_df = pd.read_csv(f'RUL_{self.dataset_name}.txt', sep=r'\s+', header=None, names=['RUL'])
        
        print(f"数据集 {self.dataset_name} 加载完成:")
        print(f"训练集形状: {self.train_df.shape}")
        print(f"测试集形状: {self.test_df.shape}")
        print(f"RUL标签形状: {self.rul_df.shape}")
        
    def preprocess_data(self):
        """数据预处理"""
        # 计算训练集RUL
        def calculate_rul(group):
            group['RUL'] = group['time_cycles'].max() - group['time_cycles']
            group['RUL'] = group['RUL'].clip(upper=self.max_rul)
            return group
        
        self.train_df = self.train_df.groupby('unit_number').apply(calculate_rul).reset_index(drop=True)
        
        # 剔除无信息传感器（标准差为0的列）
        sensor_cols = [col for col in self.train_df.columns if 'sensor' in col]
        std_values = self.train_df[sensor_cols].std()
        useful_sensors = std_values[std_values > 0].index.tolist()
        
        # 选择特征列
        self.feature_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + useful_sensors
        print(f"保留的特征列数量: {len(self.feature_cols)}")
        
        # 标准化特征
        self.train_df[self.feature_cols] = self.scaler.fit_transform(self.train_df[self.feature_cols])
        self.test_df[self.feature_cols] = self.scaler.transform(self.test_df[self.feature_cols])
        
    def extract_statistical_features(self, window):
        """提取统计特征"""
        features = []
        for col in range(window.shape[1]):
            col_data = window[:, col]
            features.extend([
                np.mean(col_data),
                np.std(col_data),
                np.max(col_data),
                np.min(col_data),
                np.ptp(col_data),  # 极差
                np.median(col_data),
                np.percentile(col_data, 25),
                np.percentile(col_data, 75),
                np.sum(np.diff(col_data) > 0) / max(len(col_data) - 1, 1),  # 上升趋势比例
                # 新增特征
                np.var(col_data),  # 方差
                np.sum(np.abs(np.diff(col_data))) / max(len(col_data) - 1, 1),  # 平均绝对变化
                col_data[-1] - col_data[0] if len(col_data) > 1 else 0,  # 总变化量
            ])
        return features
    
    def create_sliding_window(self, df, is_train=True):
        """创建滑动窗口特征"""
        X = []
        y = []
        groups = []
        
        for unit in df['unit_number'].unique():
            unit_data = df[df['unit_number'] == unit][self.feature_cols]
            unit_rul = df[df['unit_number'] == unit]['RUL'] if is_train else None
            
            if len(unit_data) < self.window_size:
                # 如果数据不足，用最后一行填充
                padding_needed = self.window_size - len(unit_data)
                last_row = unit_data.iloc[-1:] if len(unit_data) > 0 else pd.DataFrame(np.zeros((1, len(self.feature_cols))), columns=self.feature_cols)
                padding_df = pd.concat([last_row] * padding_needed, ignore_index=True)
                unit_data = pd.concat([unit_data, padding_df], ignore_index=True)
                if is_train:
                    last_rul = unit_rul.iloc[-1] if len(unit_rul) > 0 else 0
                    padding_rul = pd.Series([last_rul] * padding_needed)
                    unit_rul = pd.concat([unit_rul, padding_rul], ignore_index=True)
            
            if is_train:
                # 训练集：创建滑动窗口
                for i in range(len(unit_data) - self.window_size + 1):
                    window = unit_data.iloc[i:i+self.window_size].values
                    X.append(window)
                    y.append(unit_rul.iloc[i+self.window_size-1])
                    groups.append(unit)
            else:
                # 测试集：只取最后一个窗口
                window = unit_data.iloc[-self.window_size:].values
                X.append(window)
                groups.append(unit)
        
        X = np.array(X)
        y = np.array(y) if is_train else None
        groups = np.array(groups) if len(groups) > 0 else None

        return X, y, groups
    
    def feature_engineering(self):
        """特征工程"""
        # 为训练集创建滑动窗口
        self.X_train_seq, self.y_train, self.train_groups = self.create_sliding_window(self.train_df, is_train=True)
        
        # 提取统计特征（用于机器学习算法）
        self.X_train_stat = np.array([self.extract_statistical_features(window) for window in self.X_train_seq])
        
        # 为测试集创建滑动窗口
        self.X_test_seq, _, _ = self.create_sliding_window(self.test_df, is_train=False)
        self.y_test = self.rul_df['RUL'].clip(upper=self.max_rul).values
        
        # 提取测试集统计特征
        self.X_test_stat = np.array([self.extract_statistical_features(window) for window in self.X_test_seq])
        
        print(f"特征工程完成:")
        print(f"训练集序列特征形状: {self.X_train_seq.shape}")
        print(f"训练集统计特征形状: {self.X_train_stat.shape}")
        print(f"测试集序列特征形状: {self.X_test_seq.shape}")
        print(f"测试集统计特征形状: {self.X_test_stat.shape}")
    
    def compute_phm_score(self, y_true, y_pred):
        """计算PHM官方Score"""
        score = 0
        for true, pred in zip(y_true, y_pred):
            if pred < true:
                score += np.exp(-(true - pred) / 13) - 1
            else:
                score += np.exp((pred - true) / 10) - 1
        return score
    
    def train_random_forest(self):
        """训练随机森林模型"""
        print("训练随机森林模型...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train_stat, self.y_train)
        
        self.models['RandomForest'] = grid_search.best_estimator_
        self.best_params['RandomForest'] = grid_search.best_params_
        print(f"随机森林最优参数: {grid_search.best_params_}")
        
    def train_xgboost(self):
        """训练XGBoost模型"""
        print("训练XGBoost模型...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train_stat, self.y_train)
        
        self.models['XGBoost'] = grid_search.best_estimator_
        self.best_params['XGBoost'] = grid_search.best_params_
        print(f"XGBoost最优参数: {grid_search.best_params_}")
        
    def train_lightgbm(self):
        """训练LightGBM模型"""
        if not LIGHTGBM_AVAILABLE:
            print("跳过LightGBM模型训练 - LightGBM不可用")
            return
            
        print("训练LightGBM模型...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'num_leaves': [31, 50, 100],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        grid_search = GridSearchCV(lgb_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train_stat, self.y_train)
        
        self.models['LightGBM'] = grid_search.best_estimator_
        self.best_params['LightGBM'] = grid_search.best_params_
        print(f"LightGBM最优参数: {grid_search.best_params_}")
        
    def train_catboost(self):
        """训练CatBoost模型"""
        if not CATBOOST_AVAILABLE:
            print("跳过CatBoost模型训练 - CatBoost不可用")
            return
            
        print("训练CatBoost模型...")
        param_grid = {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [3, 6, 9],
            'l2_leaf_reg': [1, 3, 5]
        }
        
        cat_model = CatBoostRegressor(random_state=42, verbose=False)
        grid_search = GridSearchCV(cat_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train_stat, self.y_train)
        
        self.models['CatBoost'] = grid_search.best_estimator_
        self.best_params['CatBoost'] = grid_search.best_params_
        print(f"CatBoost最优参数: {grid_search.best_params_}")
        
    def train_svr(self):
        """训练支持向量回归模型"""
        print("训练SVR模型...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'epsilon': [0.01, 0.1, 0.2]
        }
        
        svr_model = SVR(kernel='rbf')
        grid_search = GridSearchCV(svr_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train_stat, self.y_train)
        
        self.models['SVR'] = grid_search.best_estimator_
        self.best_params['SVR'] = grid_search.best_params_
        print(f"SVR最优参数: {grid_search.best_params_}")
        
    def train_gradient_boosting(self):
        """训练梯度提升模型"""
        print("训练梯度提升模型...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        gb_model = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(gb_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train_stat, self.y_train)
        
        self.models['GradientBoosting'] = grid_search.best_estimator_
        self.best_params['GradientBoosting'] = grid_search.best_params_
        print(f"梯度提升最优参数: {grid_search.best_params_}")
        
    def train_extra_trees(self):
        """训练极端随机树模型"""
        print("训练极端随机树模型...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        et_model = ExtraTreesRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(et_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train_stat, self.y_train)
        
        self.models['ExtraTrees'] = grid_search.best_estimator_
        self.best_params['ExtraTrees'] = grid_search.best_params_
        print(f"极端随机树最优参数: {grid_search.best_params_}")
        
    def train_knn(self):
        """训练K近邻回归模型"""
        print("训练K近邻回归模型...")
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'p': [1, 2]
        }
        
        knn_model = KNeighborsRegressor()
        grid_search = GridSearchCV(knn_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train_stat, self.y_train)
        
        self.models['KNN'] = grid_search.best_estimator_
        self.best_params['KNN'] = grid_search.best_params_
        print(f"K近邻回归最优参数: {grid_search.best_params_}")
        
    def train_ridge(self):
        """训练Ridge回归模型"""
        print("训练Ridge回归模型...")
        param_grid = {
            'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr']
        }
        
        ridge_model = Ridge(random_state=42)
        grid_search = GridSearchCV(ridge_model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train_stat, self.y_train)
        
        self.models['Ridge'] = grid_search.best_estimator_
        self.best_params['Ridge'] = grid_search.best_params_
        print(f"Ridge回归最优参数: {grid_search.best_params_}")
        
    def build_lstm_model(self, input_shape, units=64, dropout=0.2, layers=2):
        """构建LSTM模型"""
        model = Sequential()
        
        # 第一层LSTM
        model.add(LSTM(units, return_sequences=(layers > 1), input_shape=input_shape))
        model.add(Dropout(dropout))
        
        # 额外的LSTM层
        for i in range(1, layers):
            model.add(LSTM(units//2, return_sequences=(i < layers-1)))
            model.add(Dropout(dropout))
        
        # 输出层
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
        
    def train_lstm(self):
        """训练LSTM模型"""
        if not KERAS_AVAILABLE:
            print("跳过LSTM模型训练 - Keras不可用")
            return
            
        print("训练LSTM模型...")
        
        # 参数搜索
        param_combinations = [
            {'units': 64, 'dropout': 0.2, 'layers': 2, 'batch_size': 32, 'epochs': 50},
            {'units': 128, 'dropout': 0.3, 'layers': 2, 'batch_size': 64, 'epochs': 50},
            {'units': 64, 'dropout': 0.2, 'layers': 3, 'batch_size': 32, 'epochs': 50},
        ]
        
        best_score = float('inf')
        best_model = None
        best_params = None
        
        for params in param_combinations:
            model = self.build_lstm_model(
                (self.window_size, len(self.feature_cols)),
                units=params['units'],
                dropout=params['dropout'],
                layers=params['layers']
            )
            
            # 训练模型
            history = model.fit(
                self.X_train_seq, self.y_train,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                validation_split=0.2,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                verbose=0
            )
            
            # 验证
            val_pred = model.predict(self.X_train_seq[:len(self.X_train_seq)//5])
            val_true = self.y_train[:len(self.y_train)//5]
            val_mae = mean_absolute_error(val_true, val_pred.flatten())
            
            if val_mae < best_score:
                best_score = val_mae
                best_model = model
                best_params = params
        
        self.models['LSTM'] = best_model
        self.best_params['LSTM'] = best_params
        print(f"LSTM最优参数: {best_params}")
        
    def build_gru_model(self, input_shape, units=64, dropout=0.2, layers=2):
        """构建GRU模型"""
        model = Sequential()
        
        # 第一层GRU
        model.add(GRU(units, return_sequences=(layers > 1), input_shape=input_shape))
        model.add(Dropout(dropout))
        
        # 额外的GRU层
        for i in range(1, layers):
            model.add(GRU(units//2, return_sequences=(i < layers-1)))
            model.add(Dropout(dropout))
        
        # 输出层
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
        
    def train_gru(self):
        """训练GRU模型"""
        if not KERAS_AVAILABLE:
            print("跳过GRU模型训练 - Keras不可用")
            return
            
        print("训练GRU模型...")
        
        # 参数搜索
        param_combinations = [
            {'units': 64, 'dropout': 0.2, 'layers': 2, 'batch_size': 32, 'epochs': 50},
            {'units': 128, 'dropout': 0.3, 'layers': 2, 'batch_size': 64, 'epochs': 50},
            {'units': 64, 'dropout': 0.2, 'layers': 3, 'batch_size': 32, 'epochs': 50},
        ]
        
        best_score = float('inf')
        best_model = None
        best_params = None
        
        for params in param_combinations:
            model = self.build_gru_model(
                (self.window_size, len(self.feature_cols)),
                units=params['units'],
                dropout=params['dropout'],
                layers=params['layers']
            )
            
            # 训练模型
            history = model.fit(
                self.X_train_seq, self.y_train,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                validation_split=0.2,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                verbose=0
            )
            
            # 验证
            val_pred = model.predict(self.X_train_seq[:len(self.X_train_seq)//5])
            val_true = self.y_train[:len(self.y_train)//5]
            val_mae = mean_absolute_error(val_true, val_pred.flatten())
            
            if val_mae < best_score:
                best_score = val_mae
                best_model = model
                best_params = params
        
        self.models['GRU'] = best_model
        self.best_params['GRU'] = best_params
        print(f"GRU最优参数: {best_params}")
        
    def build_cnn_lstm_model(self, input_shape, filters=64, kernel_size=3, lstm_units=64, dropout=0.2):
        """构建CNN-LSTM模型"""
        model = Sequential()
        
        # CNN层
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout))
        
        # LSTM层
        model.add(LSTM(lstm_units, return_sequences=False))
        model.add(Dropout(dropout))
        
        # 全连接层
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
        
    def train_cnn_lstm(self):
        """训练CNN-LSTM模型"""
        if not KERAS_AVAILABLE:
            print("跳过CNN-LSTM模型训练 - Keras不可用")
            return
            
        print("训练CNN-LSTM模型...")
        
        # 参数搜索
        param_combinations = [
            {'filters': 64, 'kernel_size': 3, 'lstm_units': 64, 'dropout': 0.2, 'batch_size': 32, 'epochs': 50},
            {'filters': 128, 'kernel_size': 5, 'lstm_units': 128, 'dropout': 0.3, 'batch_size': 64, 'epochs': 50},
            {'filters': 32, 'kernel_size': 3, 'lstm_units': 32, 'dropout': 0.2, 'batch_size': 32, 'epochs': 50},
        ]
        
        best_score = float('inf')
        best_model = None
        best_params = None
        
        for params in param_combinations:
            model = self.build_cnn_lstm_model(
                (self.window_size, len(self.feature_cols)),
                filters=params['filters'],
                kernel_size=params['kernel_size'],
                lstm_units=params['lstm_units'],
                dropout=params['dropout']
            )
            
            # 训练模型
            history = model.fit(
                self.X_train_seq, self.y_train,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                validation_split=0.2,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                verbose=0
            )
            
            # 验证
            val_pred = model.predict(self.X_train_seq[:len(self.X_train_seq)//5])
            val_true = self.y_train[:len(self.y_train)//5]
            val_mae = mean_absolute_error(val_true, val_pred.flatten())
            
            if val_mae < best_score:
                best_score = val_mae
                best_model = model
                best_params = params
        
        self.models['CNN-LSTM'] = best_model
        self.best_params['CNN-LSTM'] = best_params
        print(f"CNN-LSTM最优参数: {best_params}")
        
    def train_all_models(self):
        """训练所有模型"""
        print("\n开始训练所有模型...")
        
        # 机器学习模型
        self.train_random_forest()
        self.train_xgboost()
        if LIGHTGBM_AVAILABLE:
            self.train_lightgbm()
        if CATBOOST_AVAILABLE:
            self.train_catboost()
        self.train_svr()
        self.train_gradient_boosting()
        self.train_extra_trees()
        self.train_knn()
        self.train_ridge()
        
        # 深度学习模型
        if KERAS_AVAILABLE:
            self.train_lstm()
            self.train_gru()
            self.train_cnn_lstm()
        
        print(f"\n训练完成！共训练了 {len(self.models)} 个模型")
        
    def evaluate_models(self):
        """评估所有模型"""
        print("\n开始评估模型...")
        
        for name, model in self.models.items():
            # 选择合适的特征
            if name in ['LSTM', 'GRU', 'CNN-LSTM']:
                X_test = self.X_test_seq
            else:
                X_test = self.X_test_stat
            
            # 预测
            y_pred = model.predict(X_test)
            if len(y_pred.shape) > 1:
                y_pred = y_pred.flatten()
            
            # 计算指标
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            phm_score = self.compute_phm_score(self.y_test, y_pred)
            
            # 计算准确率（在±10%误差范围内的预测比例）
            accuracy = np.mean(np.abs(y_pred - self.y_test) / np.maximum(self.y_test, 1) <= 0.1) * 100
            
            self.results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'PHM_Score': phm_score,
                'Accuracy': accuracy
            }
            
            print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, PHM Score={phm_score:.4f}, Accuracy={accuracy:.2f}%")
    
    def plot_model_comparison(self):
        """绘制模型比较图"""
        if not self.results:
            print("没有结果可绘制")
            return
        
        # 创建结果DataFrame
        df_results = pd.DataFrame(self.results).T
        
        # 设置图形样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'模型性能比较 - {self.dataset_name}', fontsize=16, fontweight='bold')
        
        metrics = ['MAE', 'RMSE', 'R2', 'PHM_Score', 'Accuracy']
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
        
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            values = df_results[metric].sort_values()
            bars = ax.barh(range(len(values)), values, color=colors[i], alpha=0.7)
            
            # 添加数值标签
            for j, (idx, val) in enumerate(values.items()):
                ax.text(val + max(values) * 0.01, j, f'{val:.3f}', 
                       va='center', fontweight='bold')
            
            ax.set_yticks(range(len(values)))
            ax.set_yticklabels(values.index, fontsize=10)
            ax.set_xlabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} 比较', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # 高亮最佳值
            if metric in ['R2', 'Accuracy']:
                best_idx = len(values) - 1  # 最大值
            else:
                best_idx = 0  # 最小值
            bars[best_idx].set_color('red')
            bars[best_idx].set_alpha(0.9)
        
        # 删除多余的子图
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(f'{self.dataset_name}_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_predictions_scatter(self):
        """绘制预测值vs真实值散点图"""
        if not self.results:
            print("没有结果可绘制")
            return
        
        # 找到最佳模型（基于PHM Score）
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['PHM_Score'])
        best_model = self.models[best_model_name]
        
        # 预测
        if best_model_name in ['LSTM', 'GRU', 'CNN-LSTM']:
            y_pred = best_model.predict(self.X_test_seq).flatten()
        else:
            y_pred = best_model.predict(self.X_test_stat)
        
        # 绘制散点图
        plt.figure(figsize=(12, 10))
        
        # 主散点图
        plt.subplot(2, 2, 1)
        plt.scatter(self.y_test, y_pred, alpha=0.6, s=50)
        plt.plot([0, self.max_rul], [0, self.max_rul], 'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('真实 RUL', fontsize=12)
        plt.ylabel('预测 RUL', fontsize=12)
        plt.title(f'最佳模型预测结果 - {best_model_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 残差图
        plt.subplot(2, 2, 2)
        residuals = y_pred - self.y_test
        plt.scatter(self.y_test, residuals, alpha=0.6, s=50)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('真实 RUL', fontsize=12)
        plt.ylabel('残差 (预测值 - 真实值)', fontsize=12)
        plt.title('残差分布图', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 误差分布直方图
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('残差', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.title('残差分布直方图', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
        plt.grid(True, alpha=0.3)
        
        # 预测误差百分比
        plt.subplot(2, 2, 4)
        error_pct = np.abs(residuals) / np.maximum(self.y_test, 1) * 100
        plt.hist(error_pct, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('绝对误差百分比 (%)', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.title('预测误差百分比分布', fontsize=14, fontweight='bold')
        plt.axvline(x=10, color='r', linestyle='--', linewidth=2, label='10% 误差线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.dataset_name}_best_model_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_importance(self):
        """绘制特征重要性图"""
        # 选择有特征重要性的模型
        importance_models = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_models[name] = model.feature_importances_
        
        if not importance_models:
            print("没有可用的特征重要性信息")
            return
        
        # 创建特征名称（统计特征）
        feature_names = []
        for i, col in enumerate(self.feature_cols):
            feature_names.extend([
                f'{col}_mean', f'{col}_std', f'{col}_max', f'{col}_min',
                f'{col}_range', f'{col}_median', f'{col}_q25', f'{col}_q75',
                f'{col}_trend', f'{col}_var', f'{col}_change', f'{col}_total_change'
            ])
        
        # 绘制前几个模型的特征重要性
        n_models = min(3, len(importance_models))
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        if n_models == 1:
            axes = [axes]
        
        for i, (name, importance) in enumerate(list(importance_models.items())[:n_models]):
            # 选择前20个最重要的特征
            top_indices = np.argsort(importance)[-20:]
            top_importance = importance[top_indices]
            top_features = [feature_names[j] for j in top_indices]
            
            axes[i].barh(range(len(top_importance)), top_importance, color='skyblue', alpha=0.7)
            axes[i].set_yticks(range(len(top_importance)))
            axes[i].set_yticklabels(top_features, fontsize=8)
            axes[i].set_xlabel('重要性', fontsize=12)
            axes[i].set_title(f'{name} 特征重要性', fontsize=12, fontweight='bold')
            axes[i].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.dataset_name}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_learning_curves(self):
        """绘制学习曲线（仅适用于深度学习模型）"""
        if not KERAS_AVAILABLE:
            print("Keras不可用，跳过学习曲线绘制")
            return
        
        # 重新训练一个模型以获取历史记录
        if 'LSTM' not in self.models:
            print("没有LSTM模型可绘制学习曲线")
            return
        
        print("重新训练LSTM模型以获取学习曲线...")
        model = self.build_lstm_model((self.window_size, len(self.feature_cols)))
        
        history = model.fit(
            self.X_train_seq, self.y_train,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=0
        )
        
        # 绘制学习曲线
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失', linewidth=2)
        plt.plot(history.history['val_loss'], label='验证损失', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('损失', fontsize=12)
        plt.title('LSTM 训练损失曲线', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='训练 MAE', linewidth=2)
        plt.plot(history.history['val_mae'], label='验证 MAE', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MAE', fontsize=12)
        plt.title('LSTM MAE 曲线', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.dataset_name}_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_phm_score_analysis(self):
        """绘制PHM Score分析图"""
        if not self.results:
            print("没有结果可绘制")
            return
        
        # 获取所有模型的预测结果
        predictions = {}
        for name, model in self.models.items():
            if name in ['LSTM', 'GRU', 'CNN-LSTM']:
                y_pred = model.predict(self.X_test_seq).flatten()
            else:
                y_pred = model.predict(self.X_test_stat)
            predictions[name] = y_pred
        
        # 计算每个样本的PHM Score贡献
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. PHM Score 比较
        phm_scores = [self.results[name]['PHM_Score'] for name in self.results.keys()]
        model_names = list(self.results.keys())
        
        axes[0, 0].bar(model_names, phm_scores, color='lightcoral', alpha=0.7)
        axes[0, 0].set_ylabel('PHM Score', fontsize=12)
        axes[0, 0].set_title('各模型 PHM Score 比较', fontsize=14, fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. 最佳模型的误差分布
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['PHM_Score'])
        best_pred = predictions[best_model_name]
        errors = best_pred - self.y_test
        
        axes[0, 1].scatter(self.y_test, errors, alpha=0.6, s=30)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('真实 RUL', fontsize=12)
        axes[0, 1].set_ylabel('预测误差', fontsize=12)
        axes[0, 1].set_title(f'{best_model_name} 误差分布', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 过估计vs欠估计分析
        overestimate = errors > 0
        underestimate = errors < 0
        
        categories = ['过估计', '欠估计']
        counts = [np.sum(overestimate), np.sum(underestimate)]
        colors = ['red', 'blue']
        
        axes[1, 0].pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('预测偏向分析', fontsize=14, fontweight='bold')
        
        # 4. RUL范围内的预测精度
        rul_ranges = [(0, 25), (25, 50), (50, 75), (75, 125)]
        range_accuracies = []
        range_labels = []
        
        for low, high in rul_ranges:
            mask = (self.y_test >= low) & (self.y_test < high)
            if np.sum(mask) > 0:
                range_errors = np.abs(errors[mask])
                range_acc = np.mean(range_errors / np.maximum(self.y_test[mask], 1) <= 0.1) * 100
                range_accuracies.append(range_acc)
                range_labels.append(f'{low}-{high}')
        
        axes[1, 1].bar(range_labels, range_accuracies, color='lightgreen', alpha=0.7)
        axes[1, 1].set_ylabel('准确率 (%)', fontsize=12)
        axes[1, 1].set_xlabel('RUL 范围', fontsize=12)
        axes[1, 1].set_title('不同RUL范围的预测准确率', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.dataset_name}_phm_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_results(self):
        """保存结果到CSV文件"""
        if not self.results:
            print("没有结果可保存")
            return
        
        # 保存评估结果
        df_results = pd.DataFrame(self.results).T
        df_results.to_csv(f'{self.dataset_name}_results.csv')
        
        # 保存最优参数
        df_params = pd.DataFrame(self.best_params).T
        df_params.to_csv(f'{self.dataset_name}_best_params.csv')
        
        print(f"结果已保存到 {self.dataset_name}_results.csv")
        print(f"最优参数已保存到 {self.dataset_name}_best_params.csv")
        
    def run_complete_analysis(self):
        """运行完整分析流程"""
        print("开始完整的RUL预测分析...")
        
        # 数据处理
        self.load_data()
        self.preprocess_data()
        self.feature_engineering()
        
        # 模型训练
        self.train_all_models()
        
        # 模型评估
        self.evaluate_models()
        
        # 结果可视化
        print("\n生成可视化图表...")
        self.plot_model_comparison()
        self.plot_predictions_scatter()
        self.plot_feature_importance()
        self.plot_learning_curves()
        self.plot_phm_score_analysis()
        
        # 保存结果
        self.save_results()
        
        # 输出最佳模型
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['PHM_Score'])
        print(f"\n最佳模型: {best_model_name}")
        print(f"最佳模型性能:")
        for metric, value in self.results[best_model_name].items():
            print(f"  {metric}: {value:.4f}")
        
        return self.results, self.best_params


# 主函数
def main():
    """主函数 - 运行所有数据集的分析"""
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset}")
        print(f"{'='*60}")
        
        try:
            predictor = ComprehensiveRULPrediction(dataset_name=dataset, window_size=30, max_rul=125)
            results, params = predictor.run_complete_analysis()
            all_results[dataset] = results
        except Exception as e:
            print(f"处理数据集 {dataset} 时出错: {str(e)}")
            continue
    
    # 汇总所有数据集的结果
    if all_results:
        print(f"\n{'='*60}")
        print("所有数据集结果汇总")
        print(f"{'='*60}")
        
        summary_df = pd.DataFrame()
        for dataset, results in all_results.items():
            df = pd.DataFrame(results).T
            df['Dataset'] = dataset
            summary_df = pd.concat([summary_df, df], ignore_index=True)
        
        # 保存汇总结果
        summary_df.to_csv('all_datasets_summary.csv', index=False)
        print("汇总结果已保存到 all_datasets_summary.csv")
        
        # 绘制跨数据集比较图
        plt.figure(figsize=(15, 10))
        
        metrics = ['MAE', 'RMSE', 'R2', 'PHM_Score']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            
            # 按数据集分组绘制
            for dataset in datasets:
                if dataset in all_results:
                    dataset_data = all_results[dataset]
                    models = list(dataset_data.keys())
                    values = [dataset_data[model][metric] for model in models]
                    plt.plot(models, values, marker='o', label=dataset, linewidth=2, markersize=6)
            
            plt.xlabel('模型', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.title(f'{metric} 跨数据集比较', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cross_dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # 可以选择运行单个数据集或所有数据集
    import sys
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        print(f"运行单个数据集分析: {dataset_name}")
        predictor = ComprehensiveRULPrediction(dataset_name=dataset_name, window_size=30, max_rul=125)
        predictor.run_complete_analysis()
    else:
        print("运行所有数据集分析...")
        main()
        
    def train_