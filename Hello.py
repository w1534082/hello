import streamlit as st
# From https://github.com/argosopentech/argos-translate
import argostranslate.package
import argostranslate.translate
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()

st.write("Translator app")
st.write("You can translate into these languages: Chinese, Spanish, German, and French")

st.write("Enter text and choose your language:")

text = st.text_input("Enter input:", "")

#methods
#TASK 1
cs = ["Chinese"]#add options for spanish, german and french. You need to edit the cs list.
#after you add options the sidebar would show 4 languages
classification_space = st.sidebar.selectbox("Language to be translated into:", cs)
option = ''

if classification_space == "Chinese":
    option = 'zh'

#TASK 2
#for language codes have a look at https://cloud.google.com/translate/docs/languages
#complete the language translation option for spanish, german and french
#you have to add 4 other if clauses and in the option provide the language code
#for example for adding Spanish support, the if clause would look like this: 
#if classification_space == "Spanish"
#   option = language code for Spanish

if st.button('Translate'):
    from_code = "en"
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == option, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())
    Pronounce = argostranslate.translate.translate(text, from_code, option)
    st.write(Pronounce)
