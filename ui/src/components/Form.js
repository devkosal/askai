import React, { useState } from "react";
import { Form, Input, Rating, Button } from "semantic-ui-react";

export const Form = () => {
  const [nm, setNm] = useState("");
  const [i, setI] = useState("");
  const [ans, setAns] = useState("nothing here yet");
  const [section, setSection] = useState("nothing here yet either");

  return (
    <Form>
      <Form.Field>
        <Input
          value={nm}
          placeholder="text"
          onChange={e => setNm(e.target.value)}
        />
      </Form.Field>
      <Form.Field>
        <Input
          value={i}
          placeholder="question"
          onChange={e => setI(e.target.value)}
        />
      </Form.Field>
      <Form.Field>
        <Button
          onClick={async () => {
            const item = { nm, i };
            const response = await fetch("/", {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({
                texts: nm,
                question: i
              })
            })
              .then(response => response.json())
              .then(data => setAns(data.pred));
          }}
        >
          submit
        </Button>
      </Form.Field>
      <div>
        <p>
          {ans}
          <br />
          {section}
        </p>
      </div>
    </Form>
  );
};
