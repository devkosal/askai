import React from "react";
import { List, Header, Rating } from "semantic-ui-react";

export const Test = ({ tests }) => {
  return (
    <List>
      {tests.map(test => {
        return (
          <List.Item key={test.nm}>
            <Header>{test.nm}</Header>
            <Rating rating={test.i} maxRating={5} disabled />
          </List.Item>
        );
      })}
    </List>
  );
};
