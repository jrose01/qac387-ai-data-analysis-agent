# Tool Selection Rules

## RULES

Always check the tool registry to choose an appropriate tool first. If an appropriate tool exists, use that tool. Use code generation ONLY when an appropriate tool does not exist in th tool registry.

## Dataset overview requests
If the user asks to describe or summarize the dataset, start with a basic profile and variable-type split.

## Missing-data requests
If the user asks about missing values, use the missingness table and, if helpful, a missingness plot.

## Distribution requests
If the user asks for the distribution of a numeric variable, use a histogram.
If the user asks for the distribution of a categorical variable, use a bar chart or frequency table.

## Bivariate plots
If the user asks for a plot of two numeric variables, use a scatterplot.
If the user asks for a plot of a categorical variable and a numeric variable, use a boxplot or another appropriate plot (for example, a violin plot).

## Relationship requests
If the user asks about relationships among numeric variables, use correlations or a correlation heatmap.

## Modeling requests
If the user asks to predict or explain a numeric outcome, a multiple linear regression may be appropriate after checking assumptions and variable types.

## Safety rule
Do not choose a model before confirming that the outcome variable matches the model type.
