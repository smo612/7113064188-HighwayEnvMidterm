import tensorflow as tf
import numpy as np

# 讀取 train.npz 和 validation.npz 資料
train_data = np.load('train.npz')
train_features = train_data['data']
train_labels = train_data['label']

validation_data = np.load('validation.npz')
validation_features = validation_data['data']
validation_labels = validation_data['label']

# 創建 MirroredStrategy 用於多 GPU
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# 在 strategy 範疇內創建模型
with strategy.scope():
    # 建立改進的神經網絡模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(25,)),  # 增加神經元數量
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),  # 增加網絡層
        tf.keras.layers.Dense(10, activation='softmax')  # 假設有 10 類
    ])

    # 編譯模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # 調整學習率
                  loss='sparse_categorical_crossentropy',  # 用於分類問題
                  metrics=['accuracy'])

# 創建資料集對象
train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

validation_dataset = tf.data.Dataset.from_tensor_slices((validation_features, validation_labels))
validation_dataset = validation_dataset.batch(64)

# 訓練模型
history = model.fit(train_dataset, epochs=100, validation_data=validation_dataset)

# 訓練完成後，保存模型為 .h5 格式
model.save('trained_model.h5')

# 印出最終的驗證準確率
final_val_accuracy = history.history['val_accuracy'][-1]
print(f'Final validation accuracy: {final_val_accuracy:.4f}')