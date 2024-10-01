import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Bidirectional, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Load the dataset
data = pd.read_csv('D:/Thesis/Code/UNSW_NB15_training-set.csv')

# Feature columns and target column
feature_columns = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 
                    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 
                    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 
                    'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 
                    'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
                    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 
                    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']

target_column = 'label'

# Preprocess data
X = data[feature_columns]
y = data[target_column]

# Encode target labels as integers
y = pd.factorize(y)[0]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for Conv1D layer
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the updated model architecture
model = Sequential()

# 1st Convolutional Layer
model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1), kernel_regularizer=l2(0.001)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# 2nd Convolutional Layer
model.add(Conv1D(filters=256, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# 3rd Convolutional Layer (new)
model.add(Conv1D(filters=256, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# LSTM layer (no flattening)
model.add(Bidirectional(LSTM(128, return_sequences=True)))  # Set return_sequences=True for LSTM to output 3D data
model.add(Dropout(0.3))

# Flatten after LSTM
model.add(Flatten())

# Fully connected layers with LeakyReLU
model.add(Dense(128, kernel_regularizer=l2(0.001)))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.3))

model.add(Dense(64, kernel_regularizer=l2(0.001)))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.3))

# Output Layer (Binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model with gradient clipping
optimizer = Adam(learning_rate=0.00005, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and reducing learning rate
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Calculate class weights to address class imbalance
class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, 
          class_weight=class_weights, callbacks=[early_stopping, reduce_lr])

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
model.save('hybrid_conv_lstm_model.h5')
