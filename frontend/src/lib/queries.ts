import { useMutation, useQuery } from "@tanstack/react-query";

import {
  askInbox,
  getGmailAvailability,
  getHealth,
  listThreads,
  startGmailAuthorization,
} from "./api";

export function useThreads() {
  return useQuery({ queryKey: ["threads"], queryFn: listThreads, retry: false });
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: false,
  });
}

export function useGmailAvailability() {
  return useQuery({
    queryKey: ["gmail-availability"],
    queryFn: getGmailAvailability,
    staleTime: 5 * 60_000,
    retry: false,
  });
}

export function useAsk() {
  return useMutation({ mutationFn: askInbox });
}

export function useGmailConnect() {
  return useMutation({
    mutationFn: startGmailAuthorization,
    onSuccess: ({ authorizationUrl }) => {
      window.location.assign(authorizationUrl);
    },
  });
}
