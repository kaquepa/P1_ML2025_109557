document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("loginForm");
  if (!form) return;

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;

    // envia os dados para o Streamlit (pai do iframe)
    window.parent.postMessage(
      { type: "streamlit:setComponentValue", value: { email, password } },
      "*"
    );
  });
});
