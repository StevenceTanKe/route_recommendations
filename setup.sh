mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"stevens2002@proton.me\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml