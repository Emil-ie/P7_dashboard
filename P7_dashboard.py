#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st

import utils


def main():

    st.set_page_config(layout="wide")
    data = utils.get_data()
    st.title("Simulation d'obtention de crédit bancaire")

    col1, col2 = st.columns((1, 3))

    with col1:
        st.image(utils.show_logo(), use_column_width="auto")
    with col2:
        st.write("")
        st.header("Saisissez votre identifiant pour obtenir une réponse")
        ### Input ###
        SK_ID_CURR_user = st.selectbox("Identifiant : ", data.SK_ID_CURR.unique())

        info = utils.get_info(SK_ID_CURR_user, data)

        if not SK_ID_CURR_user:
            st.warning("Element requis")
        elif info.shape[0] == 0:
            st.warning("Nous ne trouvons pas ce numéro, veuillez réessayer")

        # import du modèle
        seuil, score, model, X_test = utils.get_model_and_score(info)

    st.markdown(" Validation du prêt")
    loanResult = "Status du prêt: "
    if score > seuil:
        loanResult += "Validé !"
        st.success(loanResult)
    else:
        loanResult += "Refusé..."
        st.error(loanResult)

    col1, col15, col2 = st.columns((1, 3, 1))

    with col1:
        st.write("")

    with col15:
        fig = utils.gauge_chart(score, 0, 1, seuil)
        st.write(fig)

    with col2:
        st.write("")

    top_features = utils.get_global_feat_imp(model, X_test)
    local = utils.get_local_feat_imp(model, X_test, info)

    ## Global & Local Features Importance
    col1, col2 = st.columns((2))
    ### Col 1/2 ### Global Features Importance
    with col1:
        st.pyplot(utils.plot_local(local))

    with col2:
        st.pyplot(utils.plot_global(top_features))


main()
# Streamlit.io pour déployer
# https://www.youtube.com/watch?v=kXvmqg8hc70
