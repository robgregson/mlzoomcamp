# ML Zoomcamp 2026 midterm project - Rob Gregson

## Data explanation

https://inseaddataanalytics.github.io/INSEADAnalytics/CourseSessions/ClassificationProcessCreditCardDefault.html

## The Business Context
A Taiwan-based credit card issuer wants to better predict the likelihood of default for its customers, as well as 
identify the key drivers that determine this likelihood. This would inform the issuer’s decisions on who to give a 
credit card to and what credit limit to provide. It would also help the issuer have a better understanding of their 
current and potential customers, which would inform their future strategy, including their planning of offering targeted 
credit products to their customers.

## The Data

Data source: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset. 
Credit to:
* Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]
* Irvine, CA: University of California, School of Information and Computer Science

The credit card issuer has gathered information on 30000 customers. The dataset contains information on 24 variables, 
including demographic factors, credit data, history of payment, and bill statements of credit card customers from April 
2005 to September 2005, as well as information on the outcome: did the customer default or not?

## Original columns:

'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
'default payment next month'

## Initial conformation

### rename the default column:
df = df.rename(columns={'default payment next month': 'default_payment_next_month'})
### rename the PAY_0 column to PAY_1 to standardise it with the others
df = df.rename(columns={'PAY_0': 'PAY_1'})

## column encodings

EDUCATION	                    (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
SEX (means gender)              (1=male, 2=female)
EDUCATION	                    (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
MARRIAGE (means marital status) (1=married, 2=single, 3=others)

The encoding of columns 7-12 (PAY_1,PAY_2,PAY_3...PAY6) is:

* -2 = no consumption, 
* -1 = pay duly, 
*  0 = the use of revolving credit, 
* 1-8 = payment delay of 1 to 8 months, 
*  9 = payment delay for nine months and above

This applies to these columns (LABEL - meaning)

PAY_1 - month 0
PAY_2 - month -1 (ie one month prior)
PAY_3 - month -2 (2 months prior)
PAY_4 - month -3 (etc.)
PAY_5 - month -4
PAY_6 - month -5

So the total number of months is 6 - the current month and 5 retrospective.

## How to score repayment history:

Create new columns:

Number of no consumption months (count number of -2 values) / total months 
Number of pay duly months (count number of -1 values) / total months 
Number of revolving credit months (count number of 0 values) / total months
Late payment rating: sum of late payment intervals (add all columns > 0) / total late payment months * max (8)

