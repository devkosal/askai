import React, { useEffect, useState } from "react";
import "./App.css";
import { Test } from "./components/Test";
import { Form } from "./components/Form";
import { Container } from "semantic-ui-react";

function App() {
  const [tests, setTests] = useState([]);

  useEffect(() => {
    fetch("/tests", {
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json"
      }
    }).then(response =>
      response.json().then(data => {
        setTests(data.tests);
      })
    );
  }, []);

  console.log(tests);

  return (
    <div className="App">
      <Container style={{ marginTop: 40 }}>
        <Form />
      </Container>
    </div>
  );
}

export default App;
