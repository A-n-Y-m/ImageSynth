<template>
  <div class="body">
    <div class="bubbles">
      <div class="bubble"></div>
      <div class="bubble"></div>
      <div class="bubble"></div>
      <div class="bubble"></div>
      <div class="bubble"></div>
      <div class="bubble"></div>
      <div class="bubble"></div>
      <div class="bubble"></div>
      <div class="bubble"></div>
      <div class="bubble"></div>
      <h1 style="padding-left: 4%;padding-top: 1%;"><span>
        <v-icon>mdi-brush-variant</v-icon>
      </span>  ImageSynth</h1>
      
      <v-container fill-height fluid class="d-flex align-center">
        <v-row align="center" justify="center" style="margin-top: 8%">
          <v-col cols="6" sm="5">
            <v-text-field
              @keydown.enter="generate"
              v-model="text"
              density="compact"
              hint="NSFW text will result in black image"
              label="Enter Text to Generate Image"
              variant="solo"
              single-line
              
            ></v-text-field>
            <v-btn-toggle
            color="light-blue"
            v-model="toggle"
            mandatory>
            <v-btn value="25" style="justify-content:center;margin-bottom: 1.1%;" >
            Low-Res
          </v-btn>
          <v-btn value="95" style="justify-content:center;margin-bottom: 1.1%;margin-left: 1.1%;" >
            High-Res
          </v-btn>
        </v-btn-toggle>
          </v-col>
          <v-btn @click="generate" style="justify-content:center;margin-bottom: 3.6%;" >
            <v-icon>mdi-play</v-icon>
          </v-btn>
          <v-col cols="6" style="padding-left: 15%">
            <v-card
              width="550"
              height="550"
              style=" background-color: #4e93e2; border-radius: 5%;"
            >
              <v-card-item>
                <v-img
                  class="pp mx-auto"
                  height="500"
                  :lazy-src="lazy_url"
                  width="500"
                  :src="url"
                  :key="url"
                  style="border-radius: 40px; margin-top: 15px"
                >
                  <div class="d-flex align-center justify-center fill-height">
                    <v-progress-circular
                      v-if="buttonPress"
                      color="grey-lighten-4"
                      indeterminate
                    ></v-progress-circular>
                  </div>
                </v-img>
              </v-card-item>
            </v-card>
          </v-col>
        </v-row>
      </v-container>
    </div>
  </div>
</template>

<script>
import axios from "axios";
import Swal from "sweetalert2";
export default {
  data() {
    return {
      toggle:1,
      text: "",
      url: "",
      buttonPress: false,
      lazy_url: "https://picsum.photos/id/11/100/60",
    };
  },
  methods: {
    generate() {
      this.buttonPress = true;
      if (this.url) {
        this.lazy_url = this.url;
        this.url = null;
      }
      var quality = this.toggle
      if(quality==1){
        quality==95
      }
      var that = this;
      axios
        .get("http://localhost:5000/data", { params: { inptext: this.text,quality:quality} })
        .then((response) => {
          if (response.data) {
            this.url = response.data;
            this.url = "data:image/png;base64," + this.url;
          }
          this.buttonPress = false;
        })
        .catch((error) => {
          Swal.fire({
            title: "Error!",
            text: "Server Offline",
            icon: "error",
            confirmButtonText: "OK",
          });
          this.buttonPress = false;
        });
    },
  },
  computed: {
    imageSrc() {
      return new URL("../assets/sample.png", import.meta.url).href;
    },
  },
};
</script>

<style>
.body {
  background-image: radial-gradient(
    #ee7752,
    #4e93e2,
    #099b46,
    rgba(209, 81, 141, 0.937)
  );
  background-size: 600% 600%;
  animation: gradient 20s ease infinite;
  height: 100vh;
}

.buttons{
  display: flex;
  align-items: flex-end;
}

.blue {
  margin-top: 50%;
}

.bubbles {
  position: relative;
  width: 100%;
  height: 100vh;
  overflow: hidden;
}

.bubbles .bubble {
  position: absolute;
  left: var(--bubble-left-offset);
  bottom: 100%;
  display: block;
  width: var(--bubble-radius);
  height: var(--bubble-radius);
  border-radius: 50%;
  animation: float-up var(--bubble-float-duration) var(--bubble-float-delay) ease-in-out;
  animation-fill-mode: forwards;
}

.bubbles .bubble::before {
  position: absolute;
  content: "";
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-image: radial-gradient(
    circle at top left,
    rgba(29, 192, 204, 0.3843137255) 40%,
    rgba(8, 186, 240, 0.4117647059) 60%,
    rgba(232, 152, 235, 0.4588235294) 90%
  );
  border-radius: inherit;
  animation: var(--bubble-sway-duration) ease-in infinite;
}

.bubbles .bubble:nth-child(0) {
  --bubble-left-offset: 62vw;
  --bubble-radius: 3vw;
  --bubble-float-duration: 5s;
  --bubble-float-delay: 3.5s;
  --bubble-sway-delay: 4s;
  --bubble-translate-offset: 31vh;
}

.bubbles .bubble:nth-child(1) {
  --bubble-left-offset: 32vw;
  --bubble-radius: 9vw;
  --bubble-float-duration: 12s;
  --bubble-float-delay: 3.5s;
  --bubble-sway-delay: 1s;
  --bubble-translate-offset: 45vh;
}

.bubbles .bubble:nth-child(2) {
  --bubble-left-offset: 79vw;
  --bubble-radius: 2vw;
  --bubble-float-duration: 4s;
  --bubble-float-delay: 1.5s;
  --bubble-sway-delay: 1s;
  --bubble-translate-offset: 87vh;
}

.bubbles .bubble:nth-child(3) {
  --bubble-left-offset: 10vw;
  --bubble-radius: 7vw;
  --bubble-float-duration: 6s;
  --bubble-float-delay: 3.5s;
  --bubble-sway-delay: 0s;
  --bubble-translate-offset: 52vh;
}

.bubbles .bubble:nth-child(4) {
  --bubble-left-offset: 68vw;
  --bubble-radius: 4vw;
  --bubble-float-duration: 5s;
  --bubble-float-delay: 3.5s;
  --bubble-sway-delay: 2s;
  --bubble-translate-offset: 10vh;
}

.bubbles .bubble:nth-child(5) {
  --bubble-left-offset: 28vw;
  --bubble-radius: 8vw;
  --bubble-float-duration: 7s;
  --bubble-float-delay: 3.5s;
  --bubble-sway-delay: 0s;
  --bubble-translate-offset: 22vh;
}

.bubbles .bubble:nth-child(6) {
  --bubble-left-offset: 52vw;
  --bubble-radius: 5vw;
  --bubble-float-duration: 4s;
  --bubble-float-delay: 3.5s;
  --bubble-sway-delay: 4s;
  --bubble-translate-offset: 98vh;
}

.bubbles .bubble:nth-child(7) {
  --bubble-left-offset: 90vw;
  --bubble-radius: 10vw;
  --bubble-float-duration: 11s;
  --bubble-float-delay: 4.5s;
  --bubble-sway-delay: 4s;
  --bubble-translate-offset: 80vh;
}

.bubbles .bubble:nth-child(8) {
  --bubble-left-offset: 5vw;
  --bubble-radius: 8vw;
  --bubble-float-duration: 6s;
  --bubble-float-delay: 2.5s;
  --bubble-sway-delay: 3s;
  --bubble-translate-offset: 78vh;
}

.bubbles .bubble:nth-child(9) {
  --bubble-left-offset: 90vw;
  --bubble-radius: 6vw;
  --bubble-float-duration: 6s;
  --bubble-float-delay: 2.5s;
  --bubble-sway-delay: 3s;
  --bubble-translate-offset: 97vh;
}

.bubbles .bubble:nth-child(10) {
  --bubble-left-offset: 26vw;
  --bubble-radius: 6vw;
  --bubble-float-duration: 6s;
  --bubble-float-delay: 2.5s;
  --bubble-sway-delay: 0s;
  --bubble-translate-offset: 39vh;
}

.bubble:hover {
  animation: moveup 0.5s ease-in;
  animation-fill-mode: forwards;
}

@keyframes moveup {
  from {
    transform: translateY(0vh);
  }

  to {
    transform: translateY(-5vh);
  }
}

@keyframes float-up {
  from {
    margin-top: 0%;
  }

  50% {
    transform: translateY(90vh);
  }

  to {
    transform: translateY(var(--bubble-translate-offset));
  }
}

@keyframes gradient {
  0% {
    background-position: 20% 20%;
  }

  25% {
    background-position: 10% 50%;
  }

  50% {
    background-position: 50% 80%;
  }

  75% {
    background-position: 90% 50%;
  }

  100% {
    background-position: 20% 20%;
  }
}

/*# sourceMappingURL=animation.css.map */
</style>
