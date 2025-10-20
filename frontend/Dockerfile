FROM node:20-alpine
WORKDIR /usr/src/app
COPY package.json package-lock.json* yarn.lock* ./
RUN npm ci || npm i
COPY . .
EXPOSE 5173
