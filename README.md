# Judging-a-book-by-its-cover

Problem : Given a dataset consisting of book cover images and book titles, the goal was to predict the genre of books (e.g., Law, Romance, etc.)

Dataset: The training data consists of around 34K cover images and book titles, with a total 
of 30 classes. The data is almost balanced, meaning each class proportion is almost similar. 
The test data has around 11k cover images and book titles distributed equally among each 
class. 

Model: Implemented different models like CNN with images, CNN with text , LSTM with images and transfomers model. Reached an accuracy of around 62% from 18% using transformers model. 

Competition Link: https://www.kaggle.com/competitions/col774-2022

Dataset Link: https://www.kaggle.com/competitions/col774-2022/data

Input and Output format

a) requirements.txt: It include the pytorch version, torchvision version etc here.
Any non-trivial libraries that are used are present here. This file would be run as 

    pip install -r requirements.txt.

b) cnn.py: Would be run as 

    python3 cnn.py <dataset dir path>
    
   This should train the CNN and generate a file <non comp test pred y.csv> containing non-competitive test set predictions.

c) rnn.py: Would be run as 

    python3 rnn.py <dataset dir path>.
    
  This should train the RNN and generate a file <non comp test pred y.csv> containing non-competitive test set predictions.

d) comp.py: Would be run as

    python3 comp.py <dataset dir path>
    
   This should train the model used in the competitive part and generate a file <comp test y.csv> containing competitive test set predictions.
