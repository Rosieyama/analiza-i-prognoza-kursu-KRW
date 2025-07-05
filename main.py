import os
import requests
import json
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def download(url, path):
    d = []

    for y in range(2010, 2025):
        response = requests.get(url.format(y=y))
        data = response.json()

        for rate in data['rates']:
            d.append(rate['mid'])
    with open(path, 'w') as f:
        json.dump(d, f)

    return d

def load(path, url):
    if os.path.exists(path):
        with open(path, 'r') as f:
            rate = json.load(f)
    else:
        rate = download(url, path)
    return rate

def window_data(data, size):
    x,y = [],[]
    for i in range(len(data)-size):
        x.append(data[i:i+size])
        y.append(data[i+size])
    return np.array(x), np.array(y)
src = 'https://api.nbp.pl/api/exchangerates/rates/a/krw/{y}-01-01/{y}-12-31/?format=json'
path = "./data.json"
print(len(load(path, src)))
df = np.array(load(path, src)).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)
X,y = window_data(scaled_data, 30)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_dim = 1
hidden_dim = 50
layer_dim = 2
output_dim = 1
lr = 0.001
num_epochs = 30

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = []
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        predictions.append(outputs)

predictions = torch.cat(predictions).numpy()
predictions = scaler.inverse_transform(predictions)

print()
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

y_test_actual = scaler.inverse_transform(y_test.numpy())

smape_value = smape(y_test_actual, predictions)
print(f'sMAPE: {smape_value:.6f}%')

#def mean_absolute_percentage_error(y_true, y_pred):
#    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#mape = mean_absolute_percentage_error(scaler.inverse_transform(y_test.numpy()), predictions)
#print(f'MAPE: {mape:.6f}%')

mae = mean_absolute_error(y_test_actual, predictions)
print(f'MAE: {mae:.6f}')

r2 = r2_score(y_test_actual, predictions)
print(f'R² Score: {r2:.6f}')

plt.figure(figsize=(10, 5))
plt.plot(scaler.inverse_transform(y_test.numpy()), label='Rzeczywiste wartości')
plt.plot(predictions, label='Przewidywania', color='red')
plt.title('Przewidywanie kursu waluty KRW na danych testowych')
plt.xlabel('Numer dnia notowania')
plt.ylabel('Kurs (PLN)')
plt.legend()
plt.show()

future_days = 757
future_predictions = []

input_seq = X_test[-1].unsqueeze(0)

for _ in range(future_days):
    with torch.no_grad():
        pred = model(input_seq).numpy()
    future_predictions.append(pred.item())

    new_input = np.append(input_seq.numpy().flatten()[1:], pred)
    input_seq = torch.tensor(new_input, dtype=torch.float32).view(1, -1, 1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

plt.figure(figsize=(10, 5))
#plt.plot(range(len(y_test_actual)), y_test_actual, label='Rzeczywiste wartości')
plt.plot(range(len(y_test_actual), len(y_test_actual) + future_days), future_predictions, label='Prognoza', color='green')
plt.title('Przewidywanie kursu KRW na 757 dni notowanych do przodu')
plt.xlabel('Numer dnia notowania')
plt.ylabel('Kurs (PLN)')
plt.legend()
plt.show()