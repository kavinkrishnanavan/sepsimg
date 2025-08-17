import main as mn

import streamlit as st

c = 0

st.write("C")

def calli():

  global c

  if c == 0:

    c = c+1
    
  else:

    mn.loginp()
  
calli()


