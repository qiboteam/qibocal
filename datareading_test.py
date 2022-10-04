from qcvv.data import Data
from qibo import models, gates

# data = Data.load_data(
#     '2022-10-03-006-jadwiga-wilkens','test', 'pickle', 'test')
data = Data.load_data(
    '2022-10-04-000-jadwiga-wilkens','standard_rb', 'pickle', 'standardrb')

print(data.df)

