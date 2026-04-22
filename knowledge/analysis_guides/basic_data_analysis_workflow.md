# Basic Data Analysis Workflow

This is a general workflow. It is not meant to be information to assist in tool calling. Refer to the tool
selection rules.md file and the files in th build0_tool_notes directory for assistance in choosing the appropriate tool.

## RULES

Always check the tool registry to choose an appropriate tool first. If an appropriate tool exists, use that tool. Use code generation ONLY when an appropriate tool does not exist in th tool registry.

## Step 1: inspect the dataset
Start by profiling the data and identifying variable types.

## Step 2: check missing data
Understand which variables have missing values and whether missingness is substantial.

## Step 3: summarize variables
Use descriptive statistics for numeric variables and frequency summaries for categorical variables.

## Step 4: visualize important distributions
Use histograms, bar charts, and missingness plots as needed.

## Step 5: examine relationships
Look at correlations or group comparisons depending on the question.

## Step 6: fit a simple model if appropriate
Only move to modeling once the agent has enough context about the data and the target variable.

## Step 7: interpret cautiously
Exploratory findings are useful, but they do not automatically imply causation or generalizability.
