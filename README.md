# CS 486 AI Project

*Click [here](https://trello.com/b/2Wdlaxlf/chessai) to go to the Trello board. To be invited to the board, click [here](https://trello.com/invite/b/2Wdlaxlf/bcad7282d063878f5bb25b9e438caea5/chessai)*

------

# Implementation Standards

## Game Data

* The raw data is in PGN format 
* The game state for machine learning (called `Tensor`) will be a `np.ndarray` with shape `(8,8,13)` (tentatively).
* For supervised network, if the labels `y = 1` means that white wins and `y = -1` means black wins. Otherwise, `y = 0` means a draw
* Labels for whether the state leads to win or lose result will be stored as a CSV file (use Pandas to read and write; learn more [here](https://www.learnpython.org/en/Pandas_Basics))
* `Tensor` will be stored in disks as Joblib files (see [here](https://joblib.readthedocs.io/en/latest/persistence.html#a-simple-example))  

## Randomization

* For all PRNG calls, set the random seed to `np.random.seed(486)`
* If you want to try different randomizations but want reproducibility, then `seed = np.random.randint(LOWER,UPPER)` and `np.random.seed(seed)` then save the `seed` somewhere if needed

## Multi-dimensional Array Indexing

* Ignoring the third dimension (piece identification), the board is a 2D array. For simplicity, suppose the board is 4x4 board. We set up arrays and index as follows:

```python
#  board = [[a,b,c,d],
#           [e,f,g,h],
#           [i,j,k,l],
#           [m,n,o,p]]

assert b == board[0, 1] == 'QUEN_B'
assert 0 == board[3, 2] == 'QUEN_W'
```

