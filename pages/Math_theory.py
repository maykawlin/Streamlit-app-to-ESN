import streamlit as st

padding_left, col, padding_right = st.columns((1, 12, 1))

with col:
    st.title('Echo State Network')
    st.header('Introduction')
    st.markdown("""
        In this article we are going to dive into the theory behind the Echo State Network (ESN).
        ESN is a very special class of recurrent neural networks (RNN). Remember that RNNs are the kind of neural
        network that stores temporal correlations of training data in their internal calculations. To be precise, a RNN
        has a hidden state, let's call it by $H_{n}$, that is given in the following way:
    """)
    
    st.latex(r'''
        H_{n} = T(H_{n - 1}, x_{n - 1})
    ''')
    
    st.markdown("""
        where $T$ is usually a nonlinear transformation, $H_{n - 1}$ is the state in the last training iteration, and x_{n - 1}
        is the last training input.
    """)