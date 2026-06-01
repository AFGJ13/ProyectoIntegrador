void setup() {

  Serial.begin(9600);

  pinMode(8, OUTPUT);   // desecho
  pinMode(9, OUTPUT);   // exportacion
  pinMode(10, OUTPUT);  // local

  digitalWrite(8, LOW);
  digitalWrite(9, LOW);
  digitalWrite(10, LOW);

  Serial.println("Arduino listo");
}

void loop() {

  if (Serial.available()) {

    char data = Serial.read();

    // Apagar todo
    digitalWrite(8, LOW);
    digitalWrite(9, LOW);
    digitalWrite(10, LOW);

    if (data == '1') {
      digitalWrite(8, HIGH);
      Serial.println("DESECHO");
    }

    else if (data == '2') {
      digitalWrite(9, HIGH);
      Serial.println("EXPORTACION");
    }

    else if (data == '3') {
      digitalWrite(10, HIGH);
      Serial.println("LOCAL");
    }
  }
}
