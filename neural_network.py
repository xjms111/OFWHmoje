import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import hashlib
import os
from typing import Tuple, List, Dict
from pathlib import Path

class HashBreakingNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int] = [512, 256, 128, 64]):
        """
        Sieć neuronowa do łamania funkcji hashujacej
        
        Args:
            input_size: Rozmiar wejścia w bitach
            output_size: Rozmiar wyjścia w bitach
            hidden_layers: Lista rozmiarów warstw ukrytych (4 rundy)
        """
        super(HashBreakingNetwork, self).__init__()
        
        if len(hidden_layers) != 4:
            raise ValueError("Sieć musi mieć dokładnie 4 rundy (warstwy ukryte)")
            
        self.input_size = input_size
        self.output_size = output_size
        
        # Definicja architektury z 4 rundami
        self.layers = nn.ModuleList()
        
        # Pierwsza runda
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.2))
        
        # Druga runda
        self.layers.append(nn.Linear(hidden_layers[0], hidden_layers[1]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.2))
        
        # Trzecia runda
        self.layers.append(nn.Linear(hidden_layers[1], hidden_layers[2]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.2))
        
        # Czwarta runda
        self.layers.append(nn.Linear(hidden_layers[2], hidden_layers[3]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.2))
        
        # Warstwa wyjściowa
        self.output_layer = nn.Linear(hidden_layers[3], output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Przepływ danych przez sieć (4 rundy)"""
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_layer(x)
        return self.sigmoid(x)

class HashBreaker:
    def __init__(self, input_bits: int = 256, output_bits: int = 256):
        """
        Klasa do trenowania i używania sieci do łamania hashów
        
        Args:
            input_bits: Długość wejścia w bitach
            output_bits: Długość wyjścia w bitach
        """
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Inicjalizacja sieci
        self.model = HashBreakingNetwork(input_bits, output_bits).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def load_training_data(self, data_folder: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ładowanie danych treningowych z folderu
        
        Args:
            data_folder: Ścieżka do folderu z danymi treningowymi
            
        Returns:
            Krotka (X, y) z danymi treningowymi
        """
        data_path = Path(data_folder)
        
        # Sprawdzenie czy folder istnieje
        if not data_path.exists():
            raise FileNotFoundError(f"Folder {data_folder} nie istnieje")
        
        # W przyszłości można dostosować do konkretnego formatu plików
        # Przykład: ładowanie z plików .npy, .csv, .pkl, etc.
        
        training_files = list(data_path.glob("*.pkl"))
        if not training_files:
            training_files = list(data_path.glob("*.npy"))
        
        if not training_files:
            raise FileNotFoundError(f"Brak plików z danymi treningowymi w {data_folder}")
        
        # Tutaj implementacja ładowania danych z plików
        # Przykład dla plików .npy:
        X_files = list(data_path.glob("*_X.npy"))
        y_files = list(data_path.glob("*_y.npy"))
        
        if X_files and y_files:
            X = np.load(X_files[0])
            y = np.load(y_files[0])
        else:
            # Domyślne dane - do usunięcia w przyszłości
            X = np.random.rand(1000, self.input_bits)
            y = np.random.rand(1000, self.output_bits)
        
        return torch.tensor(X, dtype=torch.float32, device=self.device), \
               torch.tensor(y, dtype=torch.float32, device=self.device)

    def bits_to_tensor(self, bits: str) -> torch.Tensor:
        """Konwersja ciągu bitów na tensor"""
        if len(bits) != self.input_bits:
            raise ValueError(f"Oczekiwano {self.input_bits} bitów, otrzymano {len(bits)}")
        return torch.tensor([float(bit) for bit in bits], device=self.device).unsqueeze(0)

    def tensor_to_bits(self, tensor: torch.Tensor, threshold: float = 0.5) -> str:
        """Konwersja tensora na ciąg bitów"""
        bits = ''.join(['1' if x > threshold else '0' for x in tensor.squeeze().detach().cpu().numpy()])
        if len(bits) != self.output_bits:
            bits = bits.ljust(self.output_bits, '0')[:self.output_bits]
        return bits

    def train_from_folder(self, data_folder: str, epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2):
        """Trenowanie sieci na danych z folderu"""
        # Ładowanie danych
        X, y = self.load_training_data(data_folder)
        
        # Podział na zbiór treningowy i walidacyjny
        dataset_size = len(X)
        indices = torch.randperm(dataset_size)
        split_idx = int(dataset_size * (1 - validation_split))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        print(f"Rozpoczynanie trenowania na {len(X_train)} próbkach treningowych i {len(X_val)} walidacyjnych")
        
        self.model.train()
        for epoch in range(epochs):
            # Trening
            train_loss = 0.0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Walidacja
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    batch_X = X_val[i:i+batch_size]
                    batch_y = y_val[i:i+batch_size]
                    
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            self.model.train()
            
            if epoch % 10 == 0:
                avg_train_loss = train_loss / (len(X_train) // batch_size + 1)
                avg_val_loss = val_loss / (len(X_val) // batch_size + 1)
                print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    def predict_hash(self, input_bits: str) -> str:
        """Predykcja wyjścia dla danego wejścia w postaci bitów"""
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.bits_to_tensor(input_bits)
            output_tensor = self.model(input_tensor)
            return self.tensor_to_bits(output_tensor)

    def evaluate_model(self, test_data_folder: str) -> Dict[str, float]:
        """Ewaluacja modelu na danych testowych"""
        X_test, y_test = self.load_training_data(test_data_folder)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            predictions = (outputs > 0.5).float()
            
            accuracy = (predictions == y_test).float().mean().item()
            loss = self.criterion(outputs, y_test).item()
            
            return {
                'accuracy': accuracy,
                'loss': loss,
                'total_samples': len(X_test)
            }

    def export_model(self, filepath: str):
        """Eksportowanie modelu do pliku"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_bits': self.input_bits,
            'output_bits': self.output_bits,
            'model_architecture': self.model
        }, filepath)
        print(f"Model wyeksportowany do: {filepath}")

    def load_model(self, filepath: str):
        """Ładowanie modelu z pliku"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model = checkpoint['model_architecture']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(self.device)
        print(f"Model załadowany z: {filepath}")

# Przykład użycia
if __name__ == "__main__":
    # Inicjalizacja sieci
    hash_breaker = HashBreaker(input_bits=256, output_bits=256)
    
    # Przykład trenowania na danych z folderu
    try:
        # W przyszłości: podaj ścieżkę do folderu z danymi
        data_folder = "dane_treningowe"
        hash_breaker.train_from_folder(data_folder, epochs=50, batch_size=64)
    except FileNotFoundError as e:
        print(f"Uwaga: {e}")
        print("Kontynuacja bez trenowania...")
    
    # Testowanie
    test_input = "0" * 256  # Przykładowe wejście
    try:
        predicted_output = hash_breaker.predict_hash(test_input)
        print(f"Input:  {test_input[:50]}...")
        print(f"Output: {predicted_output[:50]}...")
    except Exception as e:
        print(f"Błąd podczas predykcji: {e}")
    
    # Eksport modelu
    hash_breaker.export_model("hash_breaking_model.pth")
    
    # Przykład ewaluacji
    try:
        test_folder = "dane_testowe"
        metrics = hash_breaker.evaluate_model(test_folder)
        print(f"Wyniki ewaluacji: {metrics}")
    except FileNotFoundError:
        print("Brak danych testowych do ewaluacji")