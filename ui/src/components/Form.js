import React, { useState } from "react";
import { Form, Input, Button } from "semantic-ui-react";
const parse = require('html-react-parser');

export const QAForm = props => {
  const [question, setQuestion] = useState("");
  const [ans, setAns] = useState("");
  const [section, setSection] = useState("");
  const [qError, setQError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const config = require(`../examples/${props.example}/book-config.json`);

  const validate = e => {
    let qError = "";
    if (!question) {
      qError = "Question field cannot be blank";
    }
    if (qError) {
      setQError(qError);
      return false;
    }
    return true;
  };

  return (
    <Form>
      <Form.Field>
        <select
          className="ui dropdown"
          onChange={e => {
            setQuestion(e.target.value);
            if (e.target.value && qError) {
              setQError("");
            }
          }}
        >
          <option className="disabled item" value="">
            Select a Sample Question
          </option>
          {config.sample_questions.map((q, i) => (
            <option key={i} value={q}>
              {q}
            </option>
          ))}
        </select>
        <div className="ui horizontal divider">Or</div>
        <Input
          value={question}
          placeholder="Enter a Custom Question"
          onChange={e => {
            setQuestion(e.target.value);
            if (e.target.value && qError) {
              setQError("");
            }
          }}
        />
        <div style={{ color: "red", fontSize: 12 }}>{qError}</div>
      </Form.Field>
      <Form.Field>
        <Button
          className={isLoading ? "ui red button loading" : "ui red button"}
          onClick={async () => {
            setIsLoading(true);
            if (!validate()) {
              setIsLoading(false);
              return false;
            }
            console.log("about to begin the fetch");
            const response = await fetch("/api", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Accept: "application/json"
              },
              body: JSON.stringify({
                question: question
              })
            })
              .then(response => response.json())
              .then(data => {
                setAns(data.pred);
                setSection(data.best_section);
              });
            console.log(response);
            setIsLoading(false);
          }}
        >
          Submit
        </Button>
      </Form.Field>
      <div>
        <p style={{ fontSize: 20 }}>
          <b>{ans}</b>
        </p>
        <div className="ui segment" style={{ textAlign: "left" }}>
          <h4 className="ui header">Most relevant section:</h4>
          <p style={{ backgroundColor: "yellow" }}>{parse(`<div>${section}</div>`)}</p>
        </div>
      </div>
    </Form>
  );
};
