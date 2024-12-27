# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from SWDANmodels import SentimentDatasetSWDAN, NN2_subwordDAN, NN3_subwordDAN
from utils import BPE



# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader, epoch, lr:float=1e-4 ):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(epoch):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--random', action='store_true', help='Randomly assign values to the word embeddings')
    parser.add_argument('--emb_size', type=int, required=False, help='Embedding size', default=50)
    parser.add_argument('--hidden_ratio', type=float, required=False, help='The ratio of hidden size to emb size, also the input size', default=1)
    parser.add_argument('--emb_pretrained', type=str, required=False, help='Word Embedding Pretrained path')
    parser.add_argument('--epoch', type=int, required=False, help='Number of epochs', default=150)
    parser.add_argument('--lr', type=float, required=False, help='Learning rate', default=1e-4)
    parser.add_argument('--batchsize', type=int, required=False, help='Batch size', default=256)
    parser.add_argument('--vocab_size', type=int, required=False, help='Vocabulary size', default=1000)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load dataset
    start_time = time.time()
    # Step 1: Read the training examples
    train_examples = read_sentiment_examples('data/train.txt')
    dev_examples = read_sentiment_examples('data/dev.txt')

    # Step 2: Combine the words into a single long string
    combined_text = " ".join([" ".join(example.words) for example in train_examples])

    # Step 3: Split the combined text into a list of individual words
    training_words4BPE = combined_text.split()

    # Output: ['word1', 'word2', 'word3', ..., 'lastword']
    # training_words4BPE = [" ".join(example.words) for example in train_examples] + [" ".join(example.words) for example in dev_examples]
    # training_words4BPE = " ".join(" ".join([i.words for i in train_examples]), " ".join([i.words for i in dev_examples]))
    del train_examples, dev_examples

    tokenizer = BPE(vocab_size=args.vocab_size)
    tokenizer.train(training_words4BPE)

    train_data = SentimentDatasetSWDAN(data_path="data/train.txt", tokenizer=tokenizer)
    dev_data = SentimentDatasetSWDAN(data_path="data/dev.txt", tokenizer=tokenizer)
    train_loader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=args.batchsize, shuffle=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")
    
            # Train and evaluate NN2
    start_time = time.time()
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # train the tokenizer
    
    print('\n2 layers:')
    model_2layer = NN2_subwordDAN(input_size=args.emb_size, hidden_size=int(args.hidden_ratio*args.emb_size), vocab_size=tokenizer.vocab_size, emb_size=args.emb_size).to(device)
    nn2_train_accuracy, nn2_test_accuracy = experiment(model_2layer, train_loader, test_loader, args.epoch)

    # Train and evaluate NN3
    print('\n3 layers:')
    model_3layer = NN3_subwordDAN(input_size=args.emb_size, hidden_size=int(args.hidden_ratio*args.emb_size), vocab_size=tokenizer.vocab_size, emb_size=args.emb_size).to(device)
    nn3_train_accuracy, nn3_test_accuracy = experiment(model_3layer, train_loader, test_loader, args.epoch)

    # Plot the training accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(nn2_train_accuracy, label='2 layers')
    plt.plot(nn3_train_accuracy, label='3 layers')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy for 2, 3 Layer Networks')
    plt.legend()
    plt.grid()

    # Save the training accuracy figure
    training_accuracy_file = f'train_accuracy_{args.model}.png'
    plt.savefig(training_accuracy_file)
    print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

    # Plot the testing accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(nn2_test_accuracy, label='2 layers')
    plt.plot(nn3_test_accuracy, label='3 layers')
    plt.xlabel('Epochs')
    plt.ylabel('Dev Accuracy')
    plt.title('Dev Accuracy for 2 and 3 Layer Networks')
    plt.legend()
    plt.grid()

    # Save the testing accuracy figure
    testing_accuracy_file = f'dev_accuracy_{args.model}.png'
    plt.savefig(testing_accuracy_file)
    print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

    # plt.show()




if __name__ == "__main__":
    main()
