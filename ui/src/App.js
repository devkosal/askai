import React from "react";
import "./App.css";
import { QAForm } from "./components/Form";
import { Container } from "semantic-ui-react";

const example = process.env.REACT_APP_EXAMPLE;

const Cover = require(`./examples/${example}/cover.png`);

const config = require(`./examples/${example}/book-config.json`);

function App() {
  return (
    <div className="App">
      <Container style={{ marginTop: 40 }}>
        <div>
          <h1 className="ui header">TextbookQA</h1>
          <p>
            Welcome to TextbookQA, a question answering demo for extracting
            answers from textbooks. This demo is based on the textbook,{" "}
            <a
              target="_blank"
              rel="noopener noreferrer"
              href={`${config.book_link}`}
            >
              {`${config.book_name}`}
            </a>{" "}
            (source: openbooks). Input a respective question and receive the
            answer and the relevant section.
          </p>
          <img
            src={Cover}
            style={{ margin: 40, height: 400 }}
            alt="textbook cover"
          />
        </div>
        <QAForm example={example} />
      </Container>
    </div>
  );
}

export default App;
