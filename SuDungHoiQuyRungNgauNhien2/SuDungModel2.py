import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error


import csv
import tkinter as tk
import tkinter.ttk as ttk
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Housing")

        lbl_title = tk.Label(self, text = 'Dự báo giá nhà ở Cali',
                             font = ('Calibri', 20))
        lbl_frame = tk.LabelFrame(self)
        btn_ngau_nhien = tk.Button(lbl_frame, text = 'Lay 5 hang', 
                         width = 10, command = self.btn_ngau_nhien_click)
        btn_du_bao = tk.Button(lbl_frame, text = 'Du bao', 
                        width = 10, command = self.btn_du_bao_click)

        btn_xoa = tk.Button(lbl_frame, text = 'Xoa', width = 10, command = self.btn_xoa_click)
        btn_ngau_nhien.grid(row = 0, column = 0, padx = 5, pady = 5)
        btn_du_bao.grid(row = 1, column = 0, padx = 5, pady = 5)
        btn_xoa.grid(row = 2, column = 0, padx = 5, pady = 5)
        self.lbl_y_test = tk.Label(self, text = 'y_test = ', relief = tk.SUNKEN, 
                             bd = 1, anchor = tk.W, font = ('Consolas', 12))
        self.lbl_predict = tk.Label(self, text = 'predict = ', relief = tk.SUNKEN, 
                             bd = 1, anchor = tk.W, font = ('Consolas', 12))

        self.X_test = None 
        self.y_test = None        
        self.index = None
        self.full_pipeline = None
        self.forest_reg = None
        self.loadHousing()

        columns = ("#1", "#2", "#3", '#4', "#5", "#6", "#7", "#8", "#9")
        self.tree = ttk.Treeview(self, show="headings", columns=columns)
        self.tree.heading("#1", text="Longitude")
        self.tree.heading("#2", text="Latitude")
        self.tree.heading("#3", text="MedianAge")
        self.tree.heading("#4", text="TotalRoom")
        self.tree.heading("#5", text="TotalBedRoom")
        self.tree.heading("#6", text="Population")
        self.tree.heading("#7", text="HouseHold")
        self.tree.heading("#8", text="MedianIncome")
        self.tree.heading("#9", text="OceanProximity")

        self.tree.column("#1", anchor = tk.E, width = 70)
        self.tree.column("#2", anchor = tk.E, width = 70)
        self.tree.column("#3", anchor = tk.E, width = 70)
        self.tree.column("#4", anchor = tk.E, width = 70)
        self.tree.column("#5", anchor = tk.E, width = 90)
        self.tree.column("#6", anchor = tk.E, width = 70)
        self.tree.column("#7", anchor = tk.E, width = 70)
        self.tree.column("#8", anchor = tk.E, width = 90)
        self.tree.column("#9", anchor = tk.E, width = 110)

        ysb = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=ysb.set)
        L = len(self.X_test)
        for i in range(0, L):
            line1 = self.X_test.iloc[i]
            cot1 = line1.longitude
            cot2 = line1.latitude
            cot3 = line1.housing_median_age
            cot4 = line1.total_rooms
            cot5 = line1.total_bedrooms
            cot6 = line1.population
            cot7 = line1.households
            cot8 = line1.median_income
            cot9 = line1.ocean_proximity

            s_cot1 = '%.2f' % cot1
            s_cot2 = '%.2f' % cot2
            s_cot3 = '%d' % cot3
            s_cot4 = '%d' % cot4
            if pd.isna(cot5) == True:
                s_cot5 = ''
            else:
                s_cot5 = '%d' % cot5
            s_cot6 = '%d' % cot6
            s_cot7 = '%d' % cot7
            s_cot8 = '%.4f' % cot8

            line2 = [s_cot1, s_cot2, s_cot3, s_cot4, s_cot5, s_cot6, s_cot7, s_cot8, cot9]

            self.tree.insert("", tk.END, values = line2)


        self.tree_ngau_nhien = ttk.Treeview(self, show="headings", columns=columns, height = 5)
        self.tree_ngau_nhien.heading("#1", text="Longitude")
        self.tree_ngau_nhien.heading("#2", text="Latitude")
        self.tree_ngau_nhien.heading("#3", text="MedianAge")
        self.tree_ngau_nhien.heading("#4", text="TotalRoom")
        self.tree_ngau_nhien.heading("#5", text="TotalBedRoom")
        self.tree_ngau_nhien.heading("#6", text="Population")
        self.tree_ngau_nhien.heading("#7", text="HouseHold")
        self.tree_ngau_nhien.heading("#8", text="MedianIncome")
        self.tree_ngau_nhien.heading("#9", text="OceanProximity")

        self.tree_ngau_nhien.column("#1", anchor = tk.E, width = 70)
        self.tree_ngau_nhien.column("#2", anchor = tk.E, width = 70)
        self.tree_ngau_nhien.column("#3", anchor = tk.E, width = 70)
        self.tree_ngau_nhien.column("#4", anchor = tk.E, width = 70)
        self.tree_ngau_nhien.column("#5", anchor = tk.E, width = 90)
        self.tree_ngau_nhien.column("#6", anchor = tk.E, width = 70)
        self.tree_ngau_nhien.column("#7", anchor = tk.E, width = 70)
        self.tree_ngau_nhien.column("#8", anchor = tk.E, width = 90)
        self.tree_ngau_nhien.column("#9", anchor = tk.E, width = 110)



        lbl_title.grid(row = 0, column = 0, columnspan = 2, padx = 5, pady = 5)
        self.tree.grid(row = 1, column = 0, columnspan = 2, padx = 5, pady = 5)
        ysb.grid(row = 1, column = 2, padx = 5, pady = 5, sticky=tk.N + tk.S)
        lbl_frame.grid(row = 1, column = 3, padx = 5, pady = 5, sticky = tk.N)

        self.tree_ngau_nhien.grid(row = 2, column = 0, columnspan = 2, padx = 5, pady = 5)
        self.lbl_y_test.grid(row = 3, column = 0, columnspan = 2, padx = 5, pady = 5, sticky = tk.EW)
        self.lbl_predict.grid(row = 4, column = 0, columnspan = 2, padx = 5, pady = 5, sticky = tk.EW)

        #self.rowconfigure(0, weight=1)
        #self.columnconfigure(0, weight=1)

    def loadHousing(self):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
            def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
                self.add_bedrooms_per_room = add_bedrooms_per_room
            def fit(self, X, y=None):
                return self # nothing else to do
            def transform(self, X, y=None):
                rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
                population_per_household = X[:, population_ix] / X[:, households_ix]
                if self.add_bedrooms_per_room:
                    bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                    return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
                else:
                    return np.c_[X, rooms_per_household, population_per_household]

        housing = pd.read_csv('./SuDungHoiQuyRungNgauNhien2/housing.csv')
        housing["income_cat"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]

        # Bỏ cột income_cat ra khỏi tập train và tập test
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

        # Bỏ cột median_house_value trong tập strat_train_set
        housing = strat_train_set.drop("median_house_value", axis=1)

        housing_num = housing.drop("ocean_proximity", axis=1)

        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])

        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        self.full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

        # Lấy được những giá trị trung gian
        self.full_pipeline.fit_transform(housing)

        self.forest_reg = joblib.load("./SuDungHoiQuyRungNgauNhien2/my_model.pkl")

        self.X_test = strat_test_set.drop("median_house_value", axis=1)
        self.y_test = strat_test_set["median_house_value"].copy()

    def btn_ngau_nhien_click(self):
        for item in self.tree_ngau_nhien.get_children():
            self.tree_ngau_nhien.delete(item)

        L = len(self.X_test)
        self.index = np.random.randint(0, L-1, 5)
        for i in self.index:
            line1 = self.X_test.iloc[i]
            cot1 = line1.longitude
            cot2 = line1.latitude
            cot3 = line1.housing_median_age
            cot4 = line1.total_rooms
            cot5 = line1.total_bedrooms
            cot6 = line1.population
            cot7 = line1.households
            cot8 = line1.median_income
            cot9 = line1.ocean_proximity

            s_cot1 = '%.2f' % cot1
            s_cot2 = '%.2f' % cot2
            s_cot3 = '%d' % cot3
            s_cot4 = '%d' % cot4
            if pd.isna(cot5) == True:
                s_cot5 = ''
            else:
                s_cot5 = '%d' % cot5
            s_cot6 = '%d' % cot6
            s_cot7 = '%d' % cot7
            s_cot8 = '%.4f' % cot8

            line2 = [s_cot1, s_cot2, s_cot3, s_cot4, s_cot5, s_cot6, s_cot7, s_cot8, cot9]
            self.tree_ngau_nhien.insert("", tk.END, values = line2)
        s = 'y_test =  '
        for i in self.index:
            s = s + '%8.0f' % (self.y_test.iloc[i]) + '      '
        self.lbl_y_test.configure(text = s)
        s = 'predict = '
        self.lbl_predict.configure(text = s)

    def btn_du_bao_click(self):
        X_test_nn = self.X_test.iloc[self.index]
        X_test_nn_prepared = self.full_pipeline.transform(X_test_nn)
        y_predictions = self.forest_reg.predict(X_test_nn_prepared)
        s = 'predict = '
        for value in y_predictions:
            s = s + '%8.0f' % value + '      '
        self.lbl_predict.configure(text = s)

    def btn_xoa_click(self):
        for item in self.tree_ngau_nhien.get_children():
            self.tree_ngau_nhien.delete(item)
        self.lbl_y_test.configure(text = 'y_test =  ')
        self.lbl_predict.configure(text = 'predict = ')


if __name__ == "__main__":
    app = App()
    app.mainloop()
