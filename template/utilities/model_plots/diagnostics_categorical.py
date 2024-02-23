import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as ss

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, log_loss

def generate_confusion_matrix(dataset, actual_col, predicted_col, threshold=0.5, totals=False, percentage=False):
    """
    Generate a confusion matrix with options for row/column totals and percentages.

    Parameters:
    dataset (pd.DataFrame): Pandas DataFrame containing the actual and predicted values.
    actual_col (str): Column name for the actual binary outcomes (0 or 1).
    predicted_col (str): Column name for the predicted probabilities for the positive class.
    threshold (float, optional): Threshold for classifying a predicted probability as positive. Defaults to 0.5.
    totals (bool, optional): Whether to add row and column totals to the confusion matrix. Defaults to False.
    percentage (str or bool, optional): Whether to convert counts to percentages. If 'Row', row percentages are calculated. 
                                         If 'Column', column percentages are calculated. If True, all cells are converted to percentages of the total. 
                                         Defaults to False.

    Returns:
    DataFrame: A DataFrame representing the confusion matrix.
    """
    
    actual = dataset[actual_col]
    predicted_prob = dataset[predicted_col]
    
    # Apply threshold to predicted probabilities
    predicted = (predicted_prob >= threshold).astype(int)
    
    # Generate confusion matrix
    matrix = confusion_matrix(actual, predicted)
    
    # Create a DataFrame for better visualization
    matrix_df = pd.DataFrame(matrix, 
                             columns=['Predicted Negative', 'Predicted Positive'], 
                             index=['Actual Negative', 'Actual Positive'])
    
    # Convert to percentages if requested
    if percentage:
        if percentage == 'Row':
            matrix_df = matrix_df.div(matrix_df.sum(axis=1), axis=0) * 100
        elif percentage == 'Column':
            matrix_df = matrix_df.div(matrix_df.sum(axis=0), axis=1) * 100
        else:
            matrix_df = matrix_df.div(matrix_df.sum().sum()) * 100
        
        # Format as percentages
        matrix_df = matrix_df.applymap(lambda x: f'{x:.2f}%')
    
    # Add row and column totals if requested
    if totals:
        matrix_df['Row Total'] = matrix_df.sum(axis=1)
        matrix_df.loc['Column Total'] = matrix_df.sum(axis=0)
    
    return matrix_df



def class_metrics(dataset, actual_col, predicted_cols, threshold=0.5):
    """
    Calculate Accuracy, Specificity, Sensitivity, and Precision based on actual and predicted values, and a given threshold.

    Parameters:
    dataset (pd.DataFrame): Pandas DataFrame containing the actual and predicted values.
    actual_col (str): Column name for the actual binary outcomes (0 or 1).
    predicted_cols (str or list of str): Column name(s) for the predicted probabilities for the positive class.
    threshold (float, optional): Threshold for classifying a predicted probability as positive. Defaults to 0.5.

    Returns:
    pd.DataFrame: A DataFrame with metrics as row indices and predicted_cols as column indices if a list of predicted_cols is provided.
    """
    if isinstance(predicted_cols, str):
        predicted_cols = [predicted_cols]

    metrics_df = pd.DataFrame()

    for predicted_col in predicted_cols:
        # Apply threshold to predicted probabilities
        dataset['binary_predicted'] = (dataset[predicted_col] >= threshold).astype(int)
        
        # Generate confusion matrix
        tn, fp, fn, tp = confusion_matrix(dataset[actual_col], dataset['binary_predicted']).ravel()

        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        precision = tp / (tp + fp)

        # Remove the temporary binary_predicted column
        dataset.drop('binary_predicted', axis=1, inplace=True)

        metrics_df[predicted_col] = [accuracy, specificity, sensitivity, precision]

    metrics_df.index = ["Accuracy", "Specificity", "Sensitivity", "Precision"]

    return metrics_df


def get_roc_gini(dataset, actual_col, predicted_cols, figsize=(6,4)):
    """
    This function calculates and plots the ROC curves for one or more predicted columns, 
    and also calculates the AUC and Gini coefficient for each.
    
    Parameters:
    dataset (pandas.DataFrame): The dataset containing the actual and predicted values.
    actual_col (str): The name of the column in the dataset that contains the actual values.
    predicted_cols (str or list of str): The name(s) of the column(s) in the dataset that contains the predicted values.
    figsize (tuple): figure size
    
    Returns:
    fig (matplotlib.figure.Figure): The ROC plot.
    metrics_df (pandas.DataFrame): A DataFrame containing the AUC and Gini coefficient for each predicted column.
    """
    # Ensure predicted_cols is a list
    if isinstance(predicted_cols, str):
        predicted_cols = [predicted_cols]
    
    fig = plt.figure(figsize=figsize)
    metrics = {'AUC': [], 'Gini': []}
    
    for predicted_col in predicted_cols:
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(dataset[actual_col], dataset[predicted_col])
        
        # Calculate AUC
        roc_auc = auc(fpr, tpr)
        
        # Calculate Gini
        gini = 2*roc_auc - 1
        
        # Add to metrics
        metrics['AUC'].append(roc_auc)
        metrics['Gini'].append(gini)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{predicted_col} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(metrics, index=predicted_cols)
    
    return fig, metrics_df


def plot_pr_curve(dataset: pd.DataFrame, actual_col: str, predicted_cols: list, figsize=(6,4)):
    """
    Plots the Precision-Recall curve and calculates the area under the curve.

    Parameters:
    dataset (pd.DataFrame): The dataset containing the actual outcomes and predicted probabilities.
    actual_col (str): The name of the column in the dataset containing the actual outcomes.
    predicted_cols (list): A list of names of columns in the dataset containing the predicted probabilities.
    figsize (tuple, optional): The size of the plot. Defaults to (6,4).

    Returns:
    fig (matplotlib.figure.Figure): The figure containing the Precision-Recall curve.
    pr_auc (dict): A dictionary where the keys are the names of the predicted_cols and the values are the corresponding area under the Precision-Recall curve.
    """
    pr_auc = {}
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in predicted_cols:
        # Calculate precision, recall, and thresholds
        precision, recall, thresholds = precision_recall_curve(dataset[actual_col], dataset[col])
        
        # Calculate area under the curve
        pr_auc[col] = auc(recall, precision)
        
        # Plot Precision-Recall curve
        ax.plot(recall, precision, label=f'{col} PR curve (area = {pr_auc[col]:.2f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower right")
    ax.grid(True)
    
    return fig, pr_auc


def calculate_pseudo_r2(actual, predicted_prob):
    """
    This function calculates McFadden's and Efron's pseudo R-squared values.

    Parameters:
    actual (numpy array): A numpy array of actual outcomes.
    predicted_prob (numpy array): A numpy array of predicted probabilities.

    Returns:
    result (dict): A dictionary with McFadden's and Efron's pseudo R-squared values.
    """
    
    # Calculate the log-likelihood of the model
    log_likelihood = -log_loss(actual, predicted_prob, normalize=False)
    
    # Calculate the log-likelihood of the null model
    actual_null = np.full_like(actual, actual.mean())
    log_likelihood_null = -log_loss(actual, actual_null, normalize=False)
    
    # Calculate McFadden's R-squared
    r_squared_mcfadden = 1 - (log_likelihood / log_likelihood_null)
    
    # Calculate Efron's R-squared
    r_squared_efron = 1 - np.sum((actual - predicted_prob)**2) / np.sum((actual - actual.mean())**2)
    
    # Return the results as a dictionary
    result = {
        'r_squared_mcfadden': r_squared_mcfadden,
        'r_squared_efron': r_squared_efron
    }
    return result


# Internal functions used to transform data
def _linear(x):
    return x
def _log(x):
    return np.log(x)
def _logit(x):
    return np.log(x/(1-x))
def _exp(x):
    return np.exp(x)
def _invlogit(x):
    return 1/(1+exp(-x))

def _log1p_abs(x):
    return np.sign(x)*np.log10(1+abs(x))

_scaling_functions = {"linear": _linear, "log": _log, "logit": _logit}
_inverse_scaling_functions = {"linear": _linear, "log": _exp, "logit": _invlogit}



def act_vs_pred_plot(used_data, actual_var, pred_var, num_buckets=20,
                     with_count=True, with_CI = False,
                     lower=0.05, upper=0.95, non_nan = True):
    """
    Generates plots of actuals vs. predicted grouped by buckets of prediction percentiles. Provides option to include
        count of observations in each bucket (in case there is concentration)

    :param used_data: a single pandas DataFrame containing the y-variable and the x-variable
    :param actual_var: a single string with the name of the actual dependent variable
    :param pred_var: a single string with the name of the model prediction
    :param num_buckets: Optional integer number of buckets (quantiles) for which to group the x-variable values
    :param with_count: Optional boolean indicating whether bars representing number of observations should be plotted
        on the secondary y-axis
    :param with_CI: Boolean indicating whether approximate binomial confidence intervals should be provided
    :param lower: lower CI level
    :param upper: upper CI level
    :param non_nan: Optional boolean indicating whether to remove observations where either actual or predicted has
        missings. If False, then will count observations based on actuals
    :return: fig, axs pair
    """

    if non_nan:
        used_data = used_data.dropna(axis = 0, how = 'any', subset = [actual_var, pred_var])

    quantiled = used_data.assign(x_bin=pd.qcut(used_data[pred_var], int(num_buckets), duplicates="drop"))

    if with_CI:
        bounded_dataset = CI_dataset(quantiled, 'x_bin', actual_var, lower, upper)

    else:
        bounded_dataset = average_count_dataset(quantiled, 'x_bin', actual_var)

    plot_dataset = bounded_dataset \
        .merge(quantiled.groupby("x_bin")[pred_var].agg('mean').reset_index(), on="x_bin", how="left") \
        .rename(columns={"mean": pred_var}) \
        .sort_values(by=pred_var)

    fig, ax1 = plt.subplots()
    ax1.plot(plot_dataset[pred_var].values, plot_dataset[actual_var].values, "o")
    
    if with_CI:
        plot_dataset['lower_bound'] = plot_dataset.lower_bound.fillna(plot_dataset[actual_var])
        plot_dataset['upper_bound'] = plot_dataset.upper_bound.fillna(plot_dataset[actual_var])
        ax1.vlines(plot_dataset[pred_var].values,
                   plot_dataset["lower_bound"].values,
                   plot_dataset["upper_bound"].values,
                   color="blue", linewidths=3)
    
    ax1.set_ylabel(actual_var)
    ax1.set_xlabel(pred_var)
    axs = [ax1]

    if with_count:
        ax2 = ax1.twinx()
        ax2.vlines(plot_dataset[pred_var], 0, plot_dataset["count"], colors="lightgrey")
        ax2.set_ylabel('count', color='grey')
        ax2.tick_params('y', colors='grey')

        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.patch.set_visible(False)

        axs.append(ax2)
        
        
    # Add 45 degree line
    x = np.linspace(*ax1.get_xlim())
    ax1.plot(x, x, "g-")

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax1.set_xlabel('Prediction')
    ax1.set_ylabel('Actuals')

    # Put a legend to the right of the current axis
    fig.tight_layout()
    plt.show()
    return fig, axs, plot_dataset
    
    

def act_vs_pred_plot_multi(used_data, actual_var, pred_vars, num_buckets=20,
                           with_count=True, with_CI=False,
                           lower=0.05, upper=0.95, non_nan=True, figsize=(10, 5)):
    """
    Generates plots of actuals vs. predicted for multiple predicted variables grouped by buckets of prediction percentiles.
    Provides option to include count of observations in each bucket (in case there is concentration) and confidence intervals.

    :param used_data: a single pandas DataFrame containing the y-variable and the x-variables
    :param actual_var: a single string with the name of the actual dependent variable
    :param pred_vars: a list of strings with the names of the model predictions
    :param num_buckets: Optional integer number of buckets (quantiles) for which to group the x-variable values
    :param with_count: Optional boolean indicating whether bars representing number of observations should be plotted
        on the secondary y-axis
    :param with_CI: Boolean indicating whether approximate binomial confidence intervals should be provided
    :param lower: lower CI level
    :param upper: upper CI level
    :param non_nan: Optional boolean indicating whether to remove observations where either actual or predicted has
        missings. If False, then will count observations based on actuals
     :param figsize: Tuple of figure size in inches (width, height)
    :return: fig, axs pair
    """

    fig, ax1 = plt.subplots(figsize=figsize)
    axs = [ax1]  # Initialize list of axes

    for pred_var in pred_vars:
        if non_nan:
            used_data = used_data.dropna(axis=0, how='any', subset=[actual_var, pred_var])

        quantiled = used_data.assign(x_bin=pd.qcut(used_data[pred_var], int(num_buckets), duplicates="drop"))

        if with_CI:
            bounded_dataset = CI_dataset(quantiled, 'x_bin', actual_var, lower, upper)
        else:
            bounded_dataset = average_count_dataset(quantiled, 'x_bin', actual_var)

        plot_dataset = bounded_dataset \
            .merge(quantiled.groupby("x_bin")[pred_var].agg('mean').reset_index(), on="x_bin", how="left") \
            .rename(columns={"mean": pred_var}) \
            .sort_values(by=pred_var)

        ax1.plot(plot_dataset[pred_var].values, plot_dataset[actual_var].values, "o", label=pred_var)

        if with_CI:
            plot_dataset['lower_bound'] = plot_dataset.lower_bound.fillna(plot_dataset[actual_var])
            plot_dataset['upper_bound'] = plot_dataset.upper_bound.fillna(plot_dataset[actual_var])
            ax1.vlines(plot_dataset[pred_var].values,
                       plot_dataset["lower_bound"].values,
                       plot_dataset["upper_bound"].values,
                       label=f"CI for {pred_var}")

    ax1.set_ylabel(actual_var)
    ax1.set_xlabel("Predicted")

    if with_count:
        ax2 = ax1.twinx()
        # Note: Handling count for multiple predictions might need a different approach or might not be directly applicable.
        ax2.set_ylabel('count', color='grey')
        ax2.tick_params('y', colors='grey')
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)
        axs.append(ax2)

    # Add 45 degree line
    x = np.linspace(*ax1.get_xlim())
    ax1.plot(x, x, "g-", label="45 degree line")

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    plt.show()
    return fig, axs



def act_vs_pred_plot_segmented(used_data, actual_var, pred_var, segment, num_buckets=20, non_nan=True):
    """
    Generates plots of actuals vs. predicted grouped by buckets of prediction percentiles for each segment,
    including None as a valid segment represented by "__NONE__".

    :param used_data: a single pandas DataFrame containing the y-variable, the x-variable, and the segment variable
    :param actual_var: a single string with the name of the actual dependent variable
    :param pred_var: a single string with the name of the model prediction
    :param segment: a string representing a column in the dataset to segment data by
    :param num_buckets: Optional integer number of buckets (quantiles) for which to group the x-variable values
    :param non_nan: Optional boolean indicating whether to remove observations where either actual or predicted has
        missings. If False, then will count observations based on actuals
    :return: fig, axs, plot_datasets_dict pair
    """

    if non_nan:
        used_data = used_data.dropna(axis=0, how='any', subset=[actual_var, pred_var])
    
    # Create a temporary segment column with None values replaced by "__NONE__"
    used_data['temp_segment'] = used_data[segment].fillna('__NONE__')

    segments = used_data['temp_segment'].unique()

    fig, ax1 = plt.subplots()

    plot_datasets_dict = {}  # Initialize an empty dictionary to store plot_dataset for each segment

    for seg in segments:
        seg_data = used_data[used_data['temp_segment'] == seg]

        quantiled = seg_data.assign(x_bin=pd.qcut(seg_data[pred_var], int(num_buckets), duplicates="drop"))

        # Calculate the average of actual_var for each bin
        plot_dataset = quantiled.groupby('x_bin')[actual_var].mean().reset_index() \
            .merge(quantiled.groupby("x_bin")[pred_var].agg('mean').reset_index(), on="x_bin", how="left") \
            .rename(columns={"mean_x": actual_var, "mean_y": pred_var}) \
            .sort_values(by=pred_var)

        ax1.plot(plot_dataset[pred_var].values, plot_dataset[actual_var].values, "o", label=seg)

        plot_datasets_dict[seg] = plot_dataset  # Add the plot_dataset to the dictionary with segment as the key

    ax1.set_ylabel(actual_var)
    ax1.set_xlabel(pred_var)
    axs = [ax1]

    # Add 45 degree line
    x = np.linspace(*ax1.get_xlim())
    ax1.plot(x, x, "g-")

    ax1.legend(title=segment)

    fig.tight_layout()
    plt.show()
    return fig, axs, plot_datasets_dict  # Return the figure, axes, and the dictionary



def model_comparison_continuous(used_data, actual_var, pred_var, x_var, num_buckets=20, y_scale="logit", x_scale="linear",
                                with_count=True, with_CI = False,
                                lower=0.05, upper=0.95, non_nan = True):
    """
    Generates plots of actuals and predicted grouped by buckets of x-variable percentiles. Provides option to scale
    the y-axis and count of observations in each bucket (in case there is concentration)

    :param used_data: a single pandas DataFrame containing the y-variable and the x-variable
    :param actual_var: a single string with the name of the actual dependent variable
    :param pred_var: a single string with the name of the model prediction
    :param x_var: a single string with the name of the x-variable
    :param num_buckets: Optional integer number of buckets (quantiles) for which to group the x-variable values
    :param x_scale: Optional string of either 'linear' or 'logit' with which to scale the x-axis
    :param y_scale: Optional string of either 'linear' or 'logit' with which to scale the y-axis
    :param with_count: Optional boolean indicating whether bars representing number of observations that should be
        plotted on the secondary y-axis
    :param with_CI: Boolean indicating whether approximate binomial confidence intervals should be provided
    :param lower: lower CI level
    :param upper: upper CI level
    :param non_nan: Optional boolean indicating whether to remove observations where either actual or predicted
        has missings. If False, then will count observations based on actuals
    :return: fig, axs pair
    """

    if non_nan:
        used_data = used_data.dropna(axis = 0, how = 'any', subset = [actual_var, pred_var])

    quantiled = used_data.assign(x_bin=pd.qcut(used_data[x_var], int(num_buckets), duplicates="drop"))

    if with_CI:
        bounded_dataset = CI_dataset(quantiled, 'x_bin', actual_var, lower, upper)

    else:
        bounded_dataset = average_count_dataset(quantiled, 'x_bin', actual_var)

    plot_dataset = bounded_dataset\
        .merge(quantiled.groupby("x_bin")[pred_var].agg('mean').reset_index(), on="x_bin", how="left") \
        .rename(columns={"mean": pred_var}) \
        .merge(quantiled.groupby("x_bin")[x_var].agg('mean').reset_index(), on="x_bin", how="left") \
        .rename(columns={"mean": x_var}) \
        .sort_values(by=x_var)
    
    # Generate plots
    fig, ax1 = plt.subplots()
    act_line, = ax1.plot(plot_dataset[x_var].values, plot_dataset[actual_var].values, "bo")
    pred_line, = ax1.plot(plot_dataset[x_var].values, plot_dataset[pred_var].values, "ro-", markersize=4)
    if with_CI:
        ax1.vlines(plot_dataset[x_var].values,
                   plot_dataset["lower_bound"].values,
                   plot_dataset["upper_bound"].values,
                   color="blue", linewidths=3, zorder=30)
                   
    # Apply the right transformations to axes
    ax1.set_yscale(y_scale)
    ax1.set_xscale(x_scale)
    ax1.set_ylabel(actual_var)
    ax1.set_xlabel(x_var)
    
    axs = [ax1]

    if with_count:
        ax2 = ax1.twinx()
        ax2.vlines(plot_dataset[x_var], 0, plot_dataset["count"], colors="lightgrey")
        ax2.set_ylabel('count', color='grey')
        ax2.tick_params('y', colors='grey')
        
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.patch.set_visible(False)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax1.legend([act_line, pred_line], ['Actual', 'Predicted'], loc='center left', bbox_to_anchor=(1.15, 0.5))
    fig.tight_layout()
    plt.show()

    return fig, axs, plot_dataset
    
    

def model_comparison_continuous_multi(used_data, actual_var, pred_vars, x_var, num_buckets=20, y_scale="logit", x_scale="linear",
                                with_count=True, with_CI=False, lower=0.05, upper=0.95, non_nan=True, figsize=(10, 5)):
    """
    Generates plots of actuals and multiple predictions grouped by buckets of x-variable percentiles. Provides option to scale
    the y-axis and count of observations in each bucket (in case there is concentration)

    :param used_data: a single pandas DataFrame containing the y-variable, the x-variable, and multiple predictions
    :param actual_var: a single string with the name of the actual dependent variable
    :param pred_vars: a list of strings with the names of the model predictions
    :param x_var: a single string with the name of the x-variable
    :param num_buckets: Optional integer number of buckets (quantiles) for which to group the x-variable values
    :param x_scale: Optional string of either 'linear' or 'logit' with which to scale the x-axis
    :param y_scale: Optional string of either 'linear' or 'logit' with which to scale the y-axis
    :param with_count: Optional boolean indicating whether bars representing number of observations that should be
        plotted on the secondary y-axis
    :param with_CI: Boolean indicating whether approximate binomial confidence intervals should be provided
    :param lower: lower CI level
    :param upper: upper CI level
    :param non_nan: Optional boolean indicating whether to remove observations where either actual or any of the predicted
        has missings. If False, then will count observations based on actuals
    :param figsize: Tuple of figure size in inches (width, height)
    :return: fig, axs pair
    """

    if non_nan:
        used_data = used_data.dropna(axis=0, how='any', subset=[actual_var] + pred_vars)

    quantiled = used_data.assign(x_bin=pd.qcut(used_data[x_var], int(num_buckets), duplicates="drop"))

    if with_CI:
        bounded_dataset = CI_dataset(quantiled, 'x_bin', actual_var, lower, upper)
    else:
        bounded_dataset = average_count_dataset(quantiled, 'x_bin', actual_var)

    # Merging prediction data
    for pred_var in pred_vars:
        bounded_dataset = bounded_dataset\
            .merge(quantiled.groupby("x_bin")[pred_var].agg('mean').reset_index(), on="x_bin", how="left") \
            .rename(columns={"mean": pred_var})

    bounded_dataset = bounded_dataset\
        .merge(quantiled.groupby("x_bin")[x_var].agg('mean').reset_index(), on="x_bin", how="left") \
        .rename(columns={"mean": x_var}) \
        .sort_values(by=x_var)

    # Generate plots
    fig, ax1 = plt.subplots(figsize=figsize)
    act_line, = ax1.plot(bounded_dataset[x_var].values, bounded_dataset[actual_var].values, "bo", label='Actual')

    # Plotting each predicted variable
    pred_lines = []
    for pred_var in pred_vars:
        pred_line, = ax1.plot(bounded_dataset[x_var].values, bounded_dataset[pred_var].values, "-o", label=pred_var, markersize=4)
        pred_lines.append(pred_line)

    if with_CI:
        for pred_var in pred_vars:
            ax1.vlines(bounded_dataset[x_var].values,
                       bounded_dataset[f"{pred_var}_lower_bound"].values,
                       bounded_dataset[f"{pred_var}_upper_bound"].values,
                       color="blue", linewidths=1, zorder=30)

    # Apply the right transformations to axes
    ax1.set_yscale(y_scale)
    ax1.set_xscale(x_scale)
    ax1.set_ylabel(actual_var)
    ax1.set_xlabel(x_var)

    axs = [ax1]

    if with_count:
        ax2 = ax1.twinx()
        ax2.vlines(bounded_dataset[x_var], 0, bounded_dataset["count"], colors="lightgrey")
        ax2.set_ylabel('count', color='grey')
        ax2.tick_params('y', colors='grey')

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax1.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))
    fig.tight_layout()
    plt.show()

    return fig, axs



def model_comparison_continuous_segmented(used_data, actual_var, pred_var, x_var, segment=None, num_buckets=20, y_scale="logit", x_scale="linear", with_count=True, non_nan=True, figsize=(5,3)):
    """
    Generates plots of actuals and predicted grouped by buckets of x-variable percentiles and segments. Provides option to scale
    the y-axis and count of observations in each bucket (in case there is concentration). Also, plots count of observations
    in a separate subplot if with_count is True.

    :param used_data: a single pandas DataFrame containing the y-variable, the x-variable, and optionally a segment variable
    :param actual_var: a single string with the name of the actual dependent variable
    :param pred_var: a single string with the name of the model prediction
    :param x_var: a single string with the name of the x-variable
    :param segment: Optional string with the name of the segment column
    :param num_buckets: Optional integer number of buckets (quantiles) for which to group the x-variable values
    :param x_scale: Optional string of either 'linear' or 'logit' with which to scale the x-axis
    :param y_scale: Optional string of either 'linear' or 'logit' with which to scale the y-axis
    :param with_count: Optional boolean indicating whether a separate subplot for count of observations should be plotted
    :param non_nan: Optional boolean indicating whether to remove observations where either actual or predicted
        has missings. If False, then will count observations based on actuals
    :return: fig, axs pair, and plot_dataset dictionary
    """
    
    used_data = used_data.copy()
    
    if non_nan:
        used_data = used_data.dropna(axis=0, how='any', subset=[actual_var, pred_var, x_var] + ([segment] if segment else []))

    # Handle None in segment column
    if segment:
        used_data[segment] = used_data[segment].fillna("__NONE__")
        segments = used_data[segment].unique()
    else:
        segments = ["__ALL__"]
        used_data["__ALL__"] = "__ALL__"
        segment = "__ALL__"

    # Calculate quantiles for the x variable
    used_data['x_bin'] = pd.qcut(used_data[x_var], int(num_buckets), duplicates="drop")

    plot_data_dict = {}
    
    # Adjust subplot creation based on with_count
    ncols = 2 if with_count else 1
    fig, axs = plt.subplots(len(segments), ncols, figsize=(figsize[0] * ncols, figsize[1] * len(segments)), squeeze=False)

    for i, seg in enumerate(segments):
        seg_data = used_data[used_data[segment] == seg]
        plot_dataset = seg_data.groupby('x_bin').agg({actual_var: 'mean', pred_var: 'mean', x_var: 'mean', segment: 'size'}).reset_index().rename(columns={segment: 'count'})
        plot_data_dict[seg] = plot_dataset

        # Plotting actual vs predicted
        ax1 = axs[i, 0]
        ax1.plot(plot_dataset[x_var], plot_dataset[actual_var], "bo-", label='Actual')
        ax1.plot(plot_dataset[x_var], plot_dataset[pred_var], "ro-", label='Predicted')
        ax1.set_yscale(y_scale)
        ax1.set_xscale(x_scale)
        ax1.set_ylabel(actual_var)
        ax1.set_xlabel(x_var)
        ax1.set_title(f"Segment: {seg}")
        ax1.legend()

        # Plotting count as a bar plot if with_count is True
        if with_count:
            ax2 = axs[i, 1]
            # Calculate the width for the bars to ensure they fit nicely within the plot
            width = np.min(np.diff(plot_dataset[x_var])) * 0.8
            ax2.bar(plot_dataset[x_var], plot_dataset['count'], width=width, color='r', align='center')
            ax2.set_ylabel('Frequency')
            ax2.set_xlabel(x_var)
            ax2.set_title(f"Count Plot for Segment: {seg}")

    plt.tight_layout()
    plt.show()

    return fig, axs, plot_data_dict



def model_comparison_categorical(used_data, actual_var, pred_var, x_var, discrete=True, y_scale="logit",
                                 with_count=True, with_CI = False,
                                 lower=0.05, upper=0.95, non_nan = True):
    """
    Generates plots of actuals and predicted grouped by categorical or discrete variables, where data is grouped by
        each distinct value of the x-variable. Provides option to scale the y-axis and count of observations in each
        bucket (in case there is concentration)

    :param used_data: a single pandas DataFrame containing the y-variable and the x-variable
    :param actual_var: a single string with the name of the actual dependent variable
    :param pred_var: a single string with the name of the model prediction
    :param x_var: a single string with the name of the x-variable
    :param discrete: a single boolean indicating whether the variable should be treated as discrete (or categorical,
        if set to False). Discrete variables must be of numeric or similar type, and the plot will automatically set
        the x-axis tickmarks. The tickmarks for categorical variables will occur at every single value.
    :param y_scale: Optional string of either 'linear' or 'logit' with which to scale the y-axis
    :param with_count: Optional boolean indicating whether bars representing number of observations should be plotted
        on the secondary y-axis
    :param with_CI: Boolean indicating whether approximate binomial confidence intervals should be provided
    :param lower: Lower probability
    :param upper: Upper probability
    :param non_nan: Optional boolean indicating whether to remove observations where either actual or predicted has
        missings. If False, then will count observations based on actuals
    :return: fig, axs pair
    """

    if non_nan:
        used_data = used_data.dropna(axis = 0, how = 'any', subset = [actual_var, pred_var])

    if with_CI:
        bounded_dataset = CI_dataset(used_data, x_var, actual_var, lower, upper)

    else:
        bounded_dataset = average_count_dataset(used_data, x_var, actual_var)

    plot_dataset = bounded_dataset \
        .merge(used_data.groupby(x_var)[pred_var].agg('mean').reset_index(), on=x_var, how="left") \
        .rename(columns={"mean": pred_var}) \
        .sort_values(by=x_var)
    
    # Generate plots
    fig, ax1 = plt.subplots()
    if discrete:
        act_line, = ax1.plot(plot_dataset[x_var].values, plot_dataset[actual_var].values, "bo")
        pred_line, = ax1.plot(plot_dataset[x_var].values, plot_dataset[pred_var].values, "ro-", markersize=4)

        if with_CI:
            ax1.vlines(plot_dataset[x_var].values, plot_dataset["lower_bound"].values,
                   plot_dataset["upper_bound"].values, color="blue", linewidths=3)
    else:
        numeric_tickmarks = np.arange(0, plot_dataset.shape[0])
        act_line, = ax1.plot(numeric_tickmarks, plot_dataset[actual_var].values, "bo")
        pred_line, = ax1.plot(numeric_tickmarks, plot_dataset[pred_var].values, "ro-", markersize=4)

        if with_CI:
            ax1.vlines(numeric_tickmarks, plot_dataset["lower_bound"].values, plot_dataset["upper_bound"].values,
                       color="blue", linewidths=3)
        plt.setp(ax1.get_xticklabels(), visible=True, rotation=90)
        plt.xticks(numeric_tickmarks, plot_dataset[x_var].values, size='small')

    axs = [ax1]

    if with_count:
        ax2 = ax1.twinx()
        if discrete:
            ax2.vlines(plot_dataset[x_var], 0, plot_dataset["count"], colors="lightgrey")
        else:
            ax2.vlines(numeric_tickmarks, 0, plot_dataset["count"], colors="lightgrey")
        ax2.set_ylabel('count', color='grey')
        ax2.tick_params('y', colors='grey')

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)
        
        axs.append(ax2)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax1.legend([act_line, pred_line], ['Actual', 'Predicted'], loc='center left', bbox_to_anchor=(1.15, 0.5))

    fig.tight_layout()
    plt.show()
    return fig, axs, plot_dataset



def model_comparison_categorical_multi(used_data, actual_var, pred_vars, x_var, discrete=True, y_scale="logit",
                                 with_count=True, with_CI=False, lower=0.05, upper=0.95, non_nan=True, figsize=(10, 5)):
    """
    Generates plots of actuals and multiple predicted variables grouped by categorical or discrete variables,
    where data is grouped by each distinct value of the x-variable. Provides option to scale the y-axis and count
    of observations in each bucket (in case there is concentration)

    :param used_data: a single pandas DataFrame containing the y-variable and the x-variable
    :param actual_var: a single string with the name of the actual dependent variable
    :param pred_vars: a list of strings with the names of the model predictions
    :param x_var: a single string with the name of the x-variable
    :param discrete: a single boolean indicating whether the variable should be treated as discrete (or categorical,
        if set to False). Discrete variables must be of numeric or similar type, and the plot will automatically set
        the x-axis tickmarks. The tickmarks for categorical variables will occur at every single value.
    :param y_scale: Optional string of either 'linear' or 'logit' with which to scale the y-axis
    :param with_count: Optional boolean indicating whether bars representing number of observations should be plotted
        on the secondary y-axis
    :param with_CI: Boolean indicating whether approximate binomial confidence intervals should be provided
    :param lower: Lower probability
    :param upper: Upper probability
    :param non_nan: Optional boolean indicating whether to remove observations where either actual or predicted has
        missings. If False, then will count observations based on actuals
    :param figsize: Tuple of figure size in inches (width, height)
    :return: fig, axs pair
    """
    
    if non_nan:
        used_data = used_data.dropna(axis=0, how='any', subset=[actual_var] + pred_vars)

    if with_CI:
        bounded_dataset = CI_dataset(used_data, x_var, actual_var, lower, upper)
    else:
        bounded_dataset = average_count_dataset(used_data, x_var, actual_var)

    # Merge each predicted variable's mean with the bounded_dataset
    for pred_var in pred_vars:
        pred_data = used_data.groupby(x_var)[pred_var].agg('mean').reset_index().rename(columns={"mean": pred_var})
        bounded_dataset = bounded_dataset.merge(pred_data, on=x_var, how="left")

    bounded_dataset = bounded_dataset.sort_values(by=x_var)
    
    # Generate plots
    fig, ax1 = plt.subplots(figsize=figsize)
    act_line, = ax1.plot(bounded_dataset[x_var].values, bounded_dataset[actual_var].values, "bo", label='Actual')

    for i, pred_var in enumerate(pred_vars):
        pred_line, = ax1.plot(bounded_dataset[x_var].values, bounded_dataset[pred_var].values, "o-", markersize=4, label=pred_var)

    if with_CI:
        # Assuming CI is to be plotted for actual_var only for simplicity
        ax1.vlines(bounded_dataset[x_var].values, bounded_dataset["lower_bound"].values,
                   bounded_dataset["upper_bound"].values, color="blue", linewidths=3)

    axs = [ax1]

    if with_count:
        ax2 = ax1.twinx()
        ax2.vlines(bounded_dataset[x_var], 0, bounded_dataset["count"], colors="lightgrey")
        ax2.set_ylabel('count', color='grey')
        ax2.tick_params('y', colors='grey')

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)
        
        axs.append(ax2)

    ax1.legend(loc='center left', bbox_to_anchor=(1.15, 0.5))

    fig.tight_layout()
    plt.show()
    return fig, axs



def model_comparison_categorical_segmented(used_data, actual_var, pred_var, x_var, 
                                           segment = None, discrete=True, y_scale="logit",
                                           with_count=True, non_nan=True, figsize=(5,3)):
    """
    Modified function to generate plots of actuals and predicted grouped by categorical or discrete variables,
    with additional functionality for segmenting the data and other adjustments.

    :param used_data: pandas DataFrame containing the y-variable, x-variable, and optional segment variable
    :param actual_var: string with the name of the actual dependent variable
    :param pred_var: string with the name of the model prediction
    :param x_var: string with the name of the x-variable
    :param segment: string representing a column for segmenting the data (optional)
    :param discrete: boolean indicating whether the variable is discrete (True) or categorical (False)
    :param y_scale: string of either 'linear' or 'logit' for scaling the y-axis
    :param with_count: boolean indicating whether to plot bars for number of observations
    :param non_nan: boolean indicating whether to remove observations with missing actual or predicted values
    :return: tuple (fig, axs, plot_data_dict)
    """

    if non_nan:
        used_data = used_data.dropna(axis=0, how='any', subset=[actual_var, pred_var])

    # Handle None segment as a special case
    if segment:
        used_data[segment] = used_data[segment].fillna("__NONE__")
    else:
        used_data["__TEMP_SEGMENT__"] = "__ALL__"
        segment = "__TEMP_SEGMENT__"

    # Adjusting the handling of x_var based on the 'discrete' argument
    if discrete:
        # For discrete variables, use the exact values for grouping
        grouping_var = x_var
    else:
        # For categorical variables, ensure they are treated as distinct categories without binning
        used_data[x_var] = pd.Categorical(used_data[x_var])
        grouping_var = x_var

    plot_data_dict = {}
    segments = used_data[segment].unique()
    num_segments = len(segments)
    
    # Determine the layout of the subplots
    # Adjust subplot creation based on with_count
    ncols = 2 if with_count else 1
    fig, axs = plt.subplots(len(segments), ncols, figsize=(figsize[0] * ncols, figsize[1] * len(segments)), squeeze=False)

    for i, seg_value in enumerate(segments):
        seg_data = used_data[used_data[segment] == seg_value]
        if discrete:
            # Group by the exact values of x_var for discrete variables
            seg_data = seg_data.groupby(grouping_var).agg(
                actual_mean=(actual_var, 'mean'),
                pred_mean=(pred_var, 'mean'),
                count=(x_var, 'size')
            ).reset_index()
        else:
            # For categorical variables, simply use the categories without additional binning
            seg_data = seg_data.groupby(grouping_var).agg(
                actual_mean=(actual_var, 'mean'),
                pred_mean=(pred_var, 'mean'),
                count=(x_var, 'size')
            ).reset_index()

        # Plotting Actual vs. Predicted
        axs[i, 0].plot(seg_data[grouping_var], seg_data['actual_mean'], "o-", label=f"Actual")
        axs[i, 0].plot(seg_data[grouping_var], seg_data['pred_mean'], "ro-", label=f"Predicted")
        axs[i, 0].set_title(f'Segment: {seg_value}')
        axs[i, 0].legend()

        if with_count:
            # Plotting Count
            axs[i, 1].bar(seg_data[grouping_var], seg_data['count'], label=f"Count")
            axs[i, 1].set_title(f'Count of Observations for {seg_value}')
            axs[i, 1].legend()

        plot_data_dict[seg_value] = seg_data

    plt.tight_layout()
    plt.show()

    # Clean up temporary segment column if created
    if "__TEMP_SEGMENT__" in used_data:
        del used_data["__TEMP_SEGMENT__"]

    return fig, axs, plot_data_dict




def CI_dataset(dataset, key, column_name, lower = 0.05, upper = 0.95):
    """
    Attaches lower and upper bounds for a given column by key

    :param dataset: a single pandas DataFrame containing the y-variable and the key
    :param key: a string representing the name of the group-by key (e.g. x-variable bins)
    :param column_name: a string representing the variable name that should be summarized
    :param lower: double indicating the upper percentile of the CI
    :param upper: double indicating the lower percentile of the CI
    :return: a pandas DataFrame that contains the summarized (mean, standard deviation bands, and counts) of the
        dependent variable grouped by the key-variable
    """

    averages = dataset.groupby(key)[column_name].agg(['mean', 'count'])\
        .rename(columns={'mean': column_name}).reset_index()

    # This assumes that the actual probability is true.
    # Alternative method is to use statsmodels.stats.proportion.proportion_confint
    plot_dataset = pd.concat([
        averages,
        pd.DataFrame({"lower_bound": averages.apply(
            lambda x: ss.binom.ppf(lower, x["count"], x[column_name]) / x["count"], axis=1)}),
        pd.DataFrame({"upper_bound": averages.apply(
            lambda x: ss.binom.ppf(upper, x["count"], x[column_name]) / x["count"], axis=1)})
    ], axis=1)

    return plot_dataset


def average_count_dataset(dataset, key, column_name):
    """
    Calculates averages and counts for a given column by key

    :param dataset: a single pandas DataFrame containing the y-variable and the key
    :param key: a string representing the name of the group-by key (e.g. x-variable bins)
    :param column_name: a list of variable names that should be summarized
    :return: a pandas DataFrame that contains the summarized (mean, standard deviation bands, and counts) of the
        dependent variable grouped by the key-variable
    """

    plot_dataset = dataset.groupby(key)[column_name].agg(['mean', 'count'])\
        .reset_index() \
        .rename(columns={"mean": column_name})

    return plot_dataset