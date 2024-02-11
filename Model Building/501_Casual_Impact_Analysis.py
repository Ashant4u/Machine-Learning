##############################################################################
# Casual Impact Analysis
##############################################################################


##############################################################################
# Import required Packages
##############################################################################

from causalimpact import CausalImpact

import pandas as pd


##############################################################################
# Import and create data
##############################################################################

# import data tables

transactions = pd.read_excel("data/grocery_database.xlsx",sheet_name="transactions")
campaign_data = pd.read_excel("data/grocery_database.xlsx",sheet_name="campaign_data")

# Aggregate transactions data to customer , date level

customer_daily_sales = transactions.groupby(["customer_id","transaction_date"])["sales_cost"].sum().reset_index()


# Merge on sign up flag
    
customer_daily_sales = pd.merge(customer_daily_sales, campaign_data,how="inner",on= "customer_id")



# Pivot the data to aggregate daily sales by sign up group

casual_impact_df = customer_daily_sales.pivot_table(index = "transaction_date",
                                                    columns = "signup_flag",
                                                    values = "sales_cost",
                                                    aggfunc = "mean")


# provide a frequency for our dateTimeIndex (aviods a warning message)

# casual_impact_df.index Check frequency

casual_impact_df.index.freq ="D"



# for causal impact analysis  we need the impacted group in first column

casual_impact_df = casual_impact_df[[1,0]]



# rename columns to something more meaningful

casual_impact_df.columns = ["member","non_member"]


##############################################################################
# Apply Casual Impact
##############################################################################


pre_period = ["2020-04-01","2020-06-30"]
post_period = ["2020-07-01","2020-09-30"]


ci = CausalImpact(casual_impact_df, pre_period, post_period)

##############################################################################
# Plot the Impact
##############################################################################

ci.plot()

##############################################################################
# Extract the summary statistics & report
##############################################################################

print(ci.summary())
print(ci.summary(output= "report"))