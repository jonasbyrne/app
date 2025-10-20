FROM node:18-alpine

WORKDIR /usr/src/app

# Copiamos solo package.json primero (mejor cache)
COPY package.json ./

# Instalar deps ignorando peer deps estrictos
RUN npm install --legacy-peer-deps

# Ahora sí copiamos el código
COPY . .

EXPOSE 3000

CMD ["npm", "start"]