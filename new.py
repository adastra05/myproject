import streamlit as st 
import pandas as pd 
import numpy as np
import plotly
import plotly_express as px
from sklearn.linear_model import LinearRegression
import scipy.optimize as optimize
st.title("The Parent Ship Analysis")

uploaded_file = st.file_uploader("Please Upload your Excel Sheet", type=None, accept_multiple_files=False,label_visibility="visible")
if uploaded_file is not None:
     df = pd.read_excel(uploaded_file)
     df['L/B']= df['Length'] / df['Breadth']
     fig = px.scatter(df, x ='Length',y ='L/B' , title ='L/B ratio',trendline='ols')
     df['B/D']= df['Breadth'] / df['Depth']
     fig2 = px.scatter(df, x='Breadth', y= 'B/D',title = 'B/D ratio',trendline= 'ols')
     df['B/T']= df['Breadth'] / df['Draft']
     fig3 = px.scatter(df, x ='Breadth',y ='B/T' , title ='B/T ratio',trendline='ols')
     df['L/D']= df['Length'] / df['Depth']
     fig4 = px.scatter(df, x='Length', y= 'L/D',title = 'L/D ratio',trendline= 'ols')
     st.plotly_chart(fig)
     st.plotly_chart(fig2)
     st.plotly_chart(fig3)
     st.plotly_chart(fig4)
     Ship = df[['Length','Breadth','Depth','Draft']]
     st.dataframe(df)
     
     row = st.multiselect('Select the row to delete:', df['ship'] )
     
     if row: 
         index = df.index[df['ship'].isin(row)].tolist()
         df = df.drop(index=index)
         st.dataframe(df)
    
         LB_ratio = df['L/B'].mean()
         LD_ratio = df['L/D'].mean()
         BD_ratio = df['B/D'].mean()
         BT_ratio = df['B/T'].mean()

         average_deadweight=df['DWT'].mean()

         Required_total_Deadweight = st.number_input('Required Deadweight')
         C = 0.845
         Displacement = Required_total_Deadweight/C
#posduinies formulae
         c = 23.5
         V= 15
         Lbp = (c*((V/(2+V))**2)*((Displacement)**(1/3)))
         LBp= Lbp*0.3048
         st.write('Lbp:',LBp)

#volker's statistics 
         Lpp = (3.5+(4.5*(V/(9.8*(Displacement)**(1/3))**0.5))*((Displacement)**(1/3)))


#schnekuluth formulae 
         cc= 3.2 
         LPP = Displacement**(0.3)*(V**(0.3))*(cc)
         st.write('Lpp:',LPP)
         breadth_avg=df['Breadth'].mean()
         depth_cal=(breadth_avg-3)/1.5
         draught_cal= 0.66*depth_cal+0.9
#ayre's Formula 
         Fn= V/((9.81*LPP)**0.5)
         ccc=1.08
         Cb=(0.14/Fn)*(((Lbp/breadth_avg)+20)/26)


#form coefficients
         cb=0.7+((1/8)*np.arctan(25*(0.23-Fn)))


         cm=0.977+0.085*(Cb-0.6)
         Cwp= Cb/(0.471+0.551*Cb)
#displacment based on estimated mould
         LBTCBp =LPP*breadth_avg*draught_cal*Cb*1.025
         TotalDWT=LBTCBp*0.85
         st.title('Optimization Table')
         df =pd.DataFrame({'LBP':[LBp,LPP+5,LPP+10,LPP+15,LPP+20],})
         df['B']=df['LBP']/LB_ratio
         df['D']= df['LBP']/LD_ratio
         df['T']=df['B']/BT_ratio
         DK=V*0.514
         df['FNVGL']=DK/(9.81*df['LBP'])**0.5
         df['CBC']=0.975-0.9*(df['FNVGL']+0.02)
         df['DEL']= df['LBP']*df['B']*df['T']*df['CBC']*1.025*(1+0.006)
         df['E']=df['LBP']*(df['B']+df['T'])+0.85*df['LBP']*(df['D']-df['T'])+250
         df['PO']=(((df['DEL'])**0.567)*(V**(3.6)/1000))
         df['WS7']=0.035*(df['E']**1.36)
         df['CB1']=df['CBC']+((1-df['CBC'])*(((0.8*df['D'])-df['T'])/(3*df['T'])))
         df['WS']=df['WS7']*(1+(0.5*(df['CB1']-0.7)))+1.5
         df['DELOU']=0.38*df['LBP']*df['B']
         df['PB']=1.02*df['PO']
         df['DELP']=0.102*df['PB']+0.025
         df['DELL']= df['WS']+df['DELP']+(0.38*df['LBP']*df['B'])
         df['DWTC']= df['DEL']-df['DELL']
         st.dataframe(df)

         fig5 = px.scatter(df, x='DWTC', y='LBP', opacity=0.65,log_x=True,trendline="ols")

         st.plotly_chart(fig5)
         results = px.get_trendline_results(fig)
         st.write(results)


         import plotly.graph_objects as go
         from scipy.optimize import curve_fit
         x=df['DWTC']
         y=df['LBP']
# Define the function for the linear trendline
         params, params_covariance = optimize.curve_fit(lambda t, a, b: a*t + b, x, y)

# Extract the slope and intercept of the trendline
         slope = params[0]
         intercept = params[1]

# Create the plot
         fig6 = go.Figure()

# Add the data and the trendline
         fig6.add_trace(go.Scatter(x=x, y=y, mode='markers'))
         fig6.add_trace(go.Scatter(x=x, y=[slope*xx + intercept for xx in x], mode='lines'))

         st.plotly_chart(fig6)
         st.write(f"y = {slope}*x + {intercept}")

         REALLENGTH =  slope*TotalDWT +intercept
         Breadth= REALLENGTH/LB_ratio
         Depth = REALLENGTH/LD_ratio
         Draft= Breadth/BT_ratio
         CB = 1.2-(0.39*(16.5/(REALLENGTH)**0.5))
         CM=0.977+0.085*(CB-0.6)
         CP=CB/CM
         freeb=Depth-Draft
#freeboard correction 
         L15= REALLENGTH/15
         DELBL3=(Depth-L15)*250*(10)**(-3)
         Freeboard= (freeb-DELBL3)
         st.title('FINAL VALUES')   
         st.write('Length:',REALLENGTH)
         st.write('Breadth:',Breadth)
         st.write('Depth:',Depth)
         st.write('Draft:',Draft)
         st.write('CB:',CB)
         st.write('CM:',CM)
         st.write('CP:',CP)
         st.write('Freeboard:',Freeboard)

 

