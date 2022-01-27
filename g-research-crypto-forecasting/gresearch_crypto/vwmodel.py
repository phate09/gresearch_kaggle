import os.path
import random

from sklearn.metrics import mean_squared_error, r2_score
import subprocess
import numpy as np
import time
import logging
from shutil import copyfile


class VWShellModel():
    """Class that interfaces with VW CLI and executes all the training at once (faster) and then reports the metrics """

    def __init__(self, model_config: list, model_name: str, output_path: str) -> None:
        self.model_name = model_name
        self.model_config = model_config
        self.save_path = os.path.join(output_path, model_name)
        self.output_path = output_path

    def train_model(self,train_data_path: str, test_data_path: str, test_label_path: str,epochs=10) -> None:
        train_prediction_path = os.path.join(self.output_path, "train_predictions.txt")
        test_prediction_path = os.path.join(self.output_path, "test_predictions.txt")
        command = ["vw", "--data", train_data_path, "--final_regressor", self.save_path, "--passes", str(epochs), "--holdout_off",
                   "--kill_cache",
                   "--cache"] + self.model_config
        command += "--save_per_pass"
        logging.info(command)
        t0 = time.time()
        p = subprocess.run(command)
        t1 = time.time()
        logging.info(f"Finished training in {(t1 - t0) / 60.0:.2f} minutes")
        rmse_test, r2_test = self.test_prediction(self.save_path, test_data_path, test_label_path, test_prediction_path)
        logging.info(f"RMSE TEST: {rmse_test}| R2 TEST: {r2_test}")
        test_metrics = {
            "rmse": rmse_test,
            "r2": r2_test,
        }
        logging.info("Now printing intermediate values")
        logging.info("---------------------------------")
        only_files = [f for f in os.listdir(self.output_path) if os.path.isfile(os.path.join(self.output_path, f))]
        past_model_files = [(int(f[len(self.model_name) + 1:]), f) for f in only_files if f.startswith(self.model_name + ".")]
        past_model_files.sort()  # sort numerically
        for epoch, file in past_model_files:
            model_path = os.path.join(self.output_path, file)
            rmse_train, r2_train = self.train_prediction(model_path, train_data_path, train_prediction_path)
            rmse_test, r2_test = self.test_prediction(model_path, test_data_path, test_label_path, test_prediction_path)
            logging.info(
                f"EPOCH {epoch + 1} : RMSE TRAIN: {rmse_train}| R2 TRAIN: {r2_train} | RMSE TEST: {rmse_test}| R2 TEST: {r2_test}"
            )
            train_metrics = {
                "rmse": rmse_train,
                "r2": r2_train,
            }
            test_metrics = {
                "rmse": rmse_test,
                "r2": r2_test,
            }
        logging.info("---------------------------------")

    @staticmethod
    def train_prediction(model_path: str, train_data_path: str, train_prediction_path: str):
        """
        Generates the train prediction and return the score. The train prediction use the label provided in the train_data file
        :param model_path: the path of the model
        :param train_data_path: the path of the train data
        :param train_prediction_path: the path of the temporary file where to save the predictions
        :return: the RMSE and R2 scores
        """
        command_predictions = ["vw", "--initial_regressor", model_path, "--data", train_data_path, "--predictions", train_prediction_path, "--testonly", "--quiet"]
        p = subprocess.run(command_predictions)
        true_labels = []
        predicted_labels = []
        with open(train_data_path, "r") as fp:
            for line in fp:
                true_labels.append(float(line.split("|")[0]))  # extract the label
        with open(train_prediction_path, "r") as fp:
            for line in fp:
                predicted_labels.append(float(line))
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        rmse_test = mean_squared_error(true_labels, predicted_labels, squared=False)
        r2_test = r2_score(true_labels, predicted_labels)
        return rmse_test, r2_test

    @staticmethod
    def test_prediction(model_path: str, test_data_path: str, test_label_path: str, test_prediction_path: str):
        """
        Generates the test predictions and return the score
        :param model_path: the path of the model
        :param test_data_path: the path of the test data
        :param test_label_path: the path of the test label
        :param test_prediction_path: the path of the temporary file where to save the predictions
        :return: the RMSE and R2 scores
        """
        command_predictions = ["vw", "--initial_regressor", model_path, "--data", test_data_path, "--predictions", test_prediction_path, "--testonly", "--quiet"]
        p = subprocess.run(command_predictions)
        true_labels = []
        predicted_labels = []
        with open(test_label_path, "r") as fp:
            for line in fp:
                true_labels.append(float(line))
        with open(test_prediction_path, "r") as fp:
            for line in fp:
                predicted_labels.append(float(line))
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        rmse_test = mean_squared_error(true_labels, predicted_labels, squared=False)
        r2_test = r2_score(true_labels, predicted_labels)
        output = f"RMSE : {rmse_test} R2 : {r2_test}"
        return rmse_test, r2_test

    def save_model(self, path: str) -> None:
        copyfile(self.save_path, path)


class VWShellIndividualEpochsModel(VWShellModel):
    """
    Class that interfaces with VW CLI and executes 1 epoch at a time
    """

    def train_model(self,train_data_path: str, test_data_path: str, test_label_path: str,epochs=10) -> None:
        train_prediction_path = os.path.join(self.output_path, "train_predictions.txt")
        test_prediction_path = os.path.join(self.output_path, "test_predictions.txt")
        t0 = time.time()
        for epoch in range(epochs):
            model_path = self.save_path + '.' + str(epoch)
            if epoch == 0:
                command = ["vw", "--data", train_data_path,
                           "--final_regressor", model_path,
                           "--kill_cache",
                           "--cache",
                           "--save_resume"] + self.model_config
            else:
                prev_model_path = self.save_path + '.' + str(epoch - 1)
                command = ["vw", "--data", train_data_path,
                           "--initial_regressor", prev_model_path,
                           "--final_regressor", model_path,
                           "--kill_cache",
                           "--cache",
                           "--save_resume"] + self.model_config
            logging.info(command)
            subprocess.run(command)
            os.system(f"cp {model_path} {self.save_path}")  # updates the last model from the most recent iteration
            rmse_train, r2_train = self.train_prediction(model_path, train_data_path, train_prediction_path)
            rmse_test, r2_test = self.test_prediction(model_path, test_data_path, test_label_path, test_prediction_path)
            logging.info(
                f"EPOCH {epoch + 1} : RMSE TRAIN: {rmse_train}| R2 TRAIN: {r2_train} | RMSE TEST: {rmse_test}| R2 TEST: {r2_test}"
            )
            train_metrics = {
                "rmse": rmse_train,
                "r2": r2_train,
            }
            test_metrics = {
                "rmse": rmse_test,
                "r2": r2_test,
            }
        t1 = time.time()
        logging.info(f"Finished training in {(t1 - t0) / 60.0:.2f} minutes")
        rmse_test, r2_test = self.test_prediction(self.save_path, test_data_path, test_label_path, test_prediction_path)
        logging.info(f"RMSE TEST: {rmse_test}| R2 TEST: {r2_test}")
        test_metrics = {
            "rmse": rmse_test,
            "r2": r2_test,
        }
