import streamlit as st
import copy

def create_parameters_tab(language_content, parameters):

    if not st.session_state.running:
        col1, col2 = st.columns(2)

        st.title(language_content["parameters_title"])

        temp_parameters = copy.deepcopy(parameters)

        with col1:
            # demographics parameters
            st.header(language_content["parameters"]["demographics"]["name"])

            if not st.session_state.running:
                # demographics natural growth
                temp_parameters["demographics"]["natural_growth"] = st.text_input(language_content["parameters"]["demographics"]["natural_growth"], parameters["demographics"]["natural_growth"])

            # geographics parameters
            st.header(language_content["parameters"]["geographics"]["name"])

            if not st.session_state.running:
                # demographics sea_blue_cutoff
                temp_parameters["geographics"]["sea_blue_cutoff"] = st.text_input(language_content["parameters"]["geographics"]["sea_blue_cutoff"], parameters["geographics"]["sea_blue_cutoff"])

                # demographics fertility_per_technology_level
                st.subheader(language_content["parameters"]["geographics"]["fertility_per_technology_level"]["name"])
                for tech_stage in temp_parameters["technology"]["technological_stages"]:
                    st.text(language_content["parameters"]["geographics"]["fertility_per_technology_level"][tech_stage])
                    for soil_type in temp_parameters["geographics"]["soil_types"]:
                        temp_parameters["geographics"]["fertility_per_technology_level"][tech_stage][soil_type + "_fertility"] = st.text_input(language_content["parameters"]["geographics"]["fertility_per_technology_level"][soil_type + "_fertility"], parameters["geographics"]["fertility_per_technology_level"][tech_stage][soil_type + "_fertility"], key=language_content["parameters"]["geographics"]["fertility_per_technology_level"][soil_type + "_fertility"] + "_" + tech_stage + "_key")

        with col2:
            # culture parameters
            st.header(language_content["parameters"]["demographics"]["name"])
            
            if not st.session_state.running:
                for coef in parameters["culture"].keys():
                    temp_parameters["culture"][coef] = st.text_input(language_content["parameters"]["culture"][coef], parameters["culture"][coef])

            # technology parameters
            st.header(language_content["parameters"]["technology"]["name"])
            
            if not st.session_state.running:
                tech_keys = list(parameters["technology"].keys())
                tech_keys.remove("technological_stages")
                for coef in tech_keys:
                    # print(temp_parameters["technology"][coef])
                    temp_parameters["technology"][coef] = st.text_input(language_content["parameters"]["technology"][coef], parameters["technology"][coef])

            # politics parameters
            st.header(language_content["parameters"]["politics"]["name"])
            # demographics fertility_per_technology_level

            if not st.session_state.running:
                for i in range(len(temp_parameters["politics"])):
                    # print(language_content["parameters"]["politics"]["stages"])
                    pol_stage = temp_parameters["politics"][i]
                    st.text(language_content["parameters"]["politics"]["stages"][pol_stage['name']])
                    pol_stages_list_coefs = list(pol_stage.keys())
                    pol_stages_list_coefs.remove("name")
                    for coef in pol_stages_list_coefs:
                        # print(pol_stage[coef])
                        temp_parameters["politics"][i][coef] = st.text_input(language_content["parameters"]["politics"][coef], parameters["politics"][i][coef], key=coef + "_" + pol_stage["name"] + "_edit")

        if st.button("Save", key="save_cell_parameters"):
            st.session_state['parameters'] = copy.deepcopy(temp_parameters)